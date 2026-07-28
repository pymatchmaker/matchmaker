#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Processor related utilities

This module contains all processor related functionality.
"""

from typing import Any, Callable, Optional
import numpy as np
from dataclasses import dataclass

from matchmaker.features.audio import SAMPLE_RATE


class Processor(object):
    """
    Abstract class for a processor.

    Sits between a Stream and the queue feeding the score follower.

    Input
    -----
    A ``(data, frame_time)`` tuple where:
      - ``data`` is ``np.ndarray`` for audio buffers, or
        ``List[Tuple[mido.Message, m_time]]`` for MIDI messages.
      - ``frame_time`` (float) is the stream's nominal time for the frame.

    Output
    ------
    Either a ``(features, perf_time)`` tuple, or ``None`` while buffering
    (e.g. a chord-buffering processor still waiting for the next note).
    Most processors pass ``frame_time`` through as ``perf_time``;
    chord-buffering MIDI processors override it with the chord onset
    time (the first note's ``m_time`` of the buffered chord).

    The Stream forwards the returned tuple to the queue unchanged, so
    downstream code (`OnlineAlignment.__call__`) always sees
    ``(features, perf_time)``.
    """

    def __call__(
        self,
        data: Any,
        **kwargs,
    ) -> Any:
        """
        Parameters
        ----------
        data : Tuple[Any, float]
            ``(data, frame_time)`` tuple from the Stream.

        Returns
        -------
        output : Tuple[np.ndarray, float] or None
            ``(features, perf_time)`` or ``None`` while buffering.
        """

        raise NotImplementedError

    def reset(self):
        """
        Resets the processor, if it has any internal states.

        This method needs to be implemented in derived classes if needed.
        """
        pass


class ProcessorWrapper(Processor):
    """
    Wraps a function as a Processor class

    Parameters
    ----------
    func : Callable
        Function to be wrapped as a `Processor`.

    Attributes
    ----------
    func : Callable
        Function wrapped as a processor.
    """

    func: Callable[[Any], Any]

    def __init__(self, func: Callable[[Any], Any]) -> None:
        super().__init__()
        self.func = func

    def __call__(self, data: Any, **kwargs) -> Any:
        output = self.func(data, **kwargs)

        return output


class DummyProcessor(Processor):
    """
    Dummy sequential output processor, which always returns
    the inputs unmodified inputs
    """

    def __call__(
        self,
        data: Any,
        **kwargs,
    ) -> Any:
        return data


@dataclass
class KorzeniowskiObservation:
    """
    Observation used by the Korzeniowski particle filter.

    Parameters
    ----------
    spectrum
        Normalized spectral representation.
    active_notes
        Active notes in [0, 1].
    onset
        Onset activation in [0, 1].
    loudness
        Frame loudness in dB.
    """

    spectrum: Optional[np.ndarray] = None

    active_notes: Optional[np.ndarray] = None

    onset: float = 0.0

    onset_notes: Optional[np.ndarray] = None

    loudness: float = 0.0


@dataclass
class KorzeniowskiScoreModel:
    """
    Score representation used by the Korzeniowski particle filter.

    All expensive preprocessing is performed once during score loading.
    """

    beat_grid: np.ndarray

    onset_positions: np.ndarray

    nearest_onsets: np.ndarray

    rest_mask: np.ndarray

    templates: np.ndarray

    active_notes: Optional[list[np.ndarray]] = None


class KorzeniowskiScoreProcessor(Processor):
    """
    Offline score processor for the Korzeniowski particle filter.

    Precomputes harmonic spectral templates and lookup tables
    from a MusicXML score.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = 4096,
        beat_resolution: float = 0.1,
        num_harmonics: int = 10,
        harmonic_bandwidth: float = 15.0,
    ):

        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.beat_resolution = beat_resolution
        self.num_harmonics = num_harmonics
        self.harmonic_bandwidth = harmonic_bandwidth

        #
        # FFT frequencies
        #
        self.frequencies = np.fft.rfftfreq(
            n_fft,
            d=1 / sample_rate,
        )

        # Cache one template for every MIDI pitch.
        self.note_templates = {
            pitch: self.build_note_template(pitch)
            for pitch in range(128)
        }

    def __call__(self, score):

        note_array = score.note_array()

        beat_grid = self.build_beat_grid(
            note_array
        )

        onset_positions = np.unique(
            note_array["onset_beat"]
        )

        active_notes = self.compute_active_notes(
            beat_grid,
            note_array,
        )

        templates = self.compute_templates(
            active_notes
        )

        rest_mask = np.array(
            [
                len(notes) == 0
                for notes in active_notes
            ]
        )

        nearest_onsets = self.compute_nearest_onsets(
            beat_grid,
            onset_positions,
        )

        return KorzeniowskiScoreModel(
            beat_grid=beat_grid,
            onset_positions=onset_positions,
            nearest_onsets=nearest_onsets,
            rest_mask=rest_mask,
            templates=templates,
            active_notes=active_notes,
        )
    
    def build_beat_grid(
        self,
        note_array,
    ):

        last = np.max(
            note_array["onset_beat"]
            +
            note_array["duration_beat"]
        )

        return np.arange(
            0.0,
            last + self.beat_resolution,
            self.beat_resolution,
        )
    
    def compute_active_notes(
        self,
        beat_grid,
        note_array,
    ):

        active = []

        for beat in beat_grid:

            mask = (
                (note_array["onset_beat"] <= beat)
                &
                (
                    beat
                    <
                    note_array["onset_beat"]
                    +
                    note_array["duration_beat"]
                )
            )

            active.append(
                note_array["pitch"][mask]
            )

        return active
    
    @staticmethod
    def midi_to_frequency(
        pitch,
    ):

        return (
            440.0
            *
            2 ** ((pitch - 69) / 12)
        )
    
    def build_note_template(
        self,
        pitch: int,
    ) -> np.ndarray:
        """
        Construct the harmonic spectral template for a
        single MIDI pitch.

        Implements the template described in
        Section 2.4 of Korzeniowski et al.
        """

        template = np.zeros_like(
            self.frequencies,
            dtype=np.float32,
        )

        f0 = self.midi_to_frequency(pitch)

        for harmonic in range(
            1,
            self.num_harmonics + 1,
        ):

            frequency = harmonic * f0

            # Ignore harmonics above Nyquist.
            if frequency > self.sample_rate / 2:
                break

            amplitude = 1.0 / (harmonic ** 2)

            sigma = self.harmonic_bandwidth * harmonic

            gaussian = np.exp(
                -0.5
                * (
                    (self.frequencies - frequency)
                    / sigma
                ) ** 2
            )

            template += amplitude * gaussian

        # Normalize.
        norm = np.linalg.norm(template)

        if norm > 0:
            template /= norm

        return template
    
    def compute_templates(
        self,
        active_notes,
    ):

        templates = np.zeros(
            (
                len(active_notes),
                len(self.frequencies),
            ),
            dtype=np.float32,
        )

        for i, notes in enumerate(
            active_notes
        ):

            if len(notes) == 0:
                continue

            for pitch in notes:

                templates[i] += self.note_templates[
                    int(pitch)
                ]

            norm = np.linalg.norm(
                templates[i]
            )

            if norm > 0:

                templates[i] /= norm

        return templates
    

    def template(
        self,
        beat,
    ):

        position = (
            beat
            / self.beat_resolution
        )

        i0 = int(np.floor(position))

        i0 = np.clip(
            i0,
            0,
            len(self.templates) - 1,
        )

        i1 = min(
            i0 + 1,
            len(self.templates) - 1,
        )

        alpha = position - i0

        template = (
            (1.0 - alpha)
            * self.templates[i0]
            +
            alpha
            * self.templates[i1]
        )

        norm = np.linalg.norm(
            template
        )

        if norm > 0:

            template /= norm

        return template

if __name__ == "__main__":
    pass
