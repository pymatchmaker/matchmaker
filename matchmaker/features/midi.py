#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains methods to compute features from MIDI signals.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import partitura as pt
from numpy.typing import NDArray
from partitura.performance import PerformanceLike, PerformedPart
from partitura.score import Part, Score, ScoreLike, merge_parts
from partitura.utils.music import performance_from_part

from matchmaker.features.processor import Processor
from matchmaker.utils.symbolic import (
    framed_midi_messages_from_performance,
    midi_messages_from_performance,
)
from matchmaker.utils.typing import InputMIDIFrame, NDArrayFloat

CHORD_THRESHOLD: float = 0.035  # seconds; matches IOI_THRESHOLD in outer_product_hmm


class PitchProcessor(Processor):
    """Aggregate ``note_on`` events in a MIDI frame into a pitch observation.

    All concurrent ``note_on`` events present in the input frame are merged
    into a single observation (i.e. a frame containing a chord emits one
    observation covering all chord pitches). Frame-level time grouping is
    the ``MidiStream``'s responsibility via ``polling_period``; this
    processor itself is stateless.

    Parameters
    ----------
    piano_range : bool, default False
        If True, restrict pitch range to the 88-key piano (MIDI 21-108).
        Otherwise use the full 128-pitch range.
    return_pitch_list : bool, default False
        Output format selector (see "Returns" for shapes).
        - False: one-hot binary pianoroll vector. Use this for observation
          models that expect Bernoulli-style binary observations (e.g.
          ``BernoulliPitchObservationModel`` used by ``PitchHMM`` /
          ``OuterProductHMM``).
        - True: 1-D array of MIDI pitch indices. Use this for observation
          models that index into pitch profiles by pitch value (e.g.
          ``ACCPitchIOIObservationModel`` used by ``PitchIOIHMM``).

    Returns
    -------
    None if the frame has no note_on events. Otherwise a tuple
    ``(pitch_obs, f_time)`` where:
        - ``pitch_obs``: pitch observation (see ``return_pitch_list``).
          * If ``return_pitch_list=False``: ``np.ndarray`` of shape
            ``(88,)`` when ``piano_range=True`` else ``(128,)``, binary
            (0/1, float32). E.g. chord (C4, E4, G4) =
            ``[..., 1 at 60, ..., 1 at 64, ..., 1 at 67, ...]`` (or the
            shifted piano-range equivalent).
          * If ``return_pitch_list=True``: ``np.ndarray`` of shape
            ``(n_notes,)`` with MIDI pitch indices (float32). E.g. chord
            (C4, E4, G4) with ``piano_range=False`` = ``[60., 64., 67.]``,
            and with ``piano_range=True`` = ``[39., 43., 46.]`` (=
            pitch - 21).
        - ``f_time``: frame timestamp from the stream (window-center
          time in windowed mode, message arrival time in single-message
          mode).
    """

    piano_range: bool

    def __init__(
        self,
        piano_range: bool = False,
        return_pitch_list: bool = False,
    ) -> None:
        super().__init__()
        self.piano_range = piano_range
        self.return_pitch_list = return_pitch_list
        self.piano_shift = 21 if piano_range else 0

    def __call__(
        self,
        frame: InputMIDIFrame,
    ) -> Optional[Tuple[NDArrayFloat, float]]:
        data, f_time = frame
        # pitch_obs = []
        pitch_obs = np.zeros(
            128,
            dtype=np.float32,
        )

        # TODO: Replace the for loop with list comprehension
        pitch_obs_list = []

        for msg, _ in data:
            if (
                getattr(msg, "type", "other") == "note_on"
                and getattr(msg, "velocity", 0) > 0
            ):
                pitch_obs[msg.note] = 1
                pitch_obs_list.append(msg.note - self.piano_shift)

        if pitch_obs.sum() > 0:
            if self.piano_range:
                pitch_obs = pitch_obs[21:109]

            if self.return_pitch_list:
                return np.array(pitch_obs_list, dtype=np.float32), f_time
            return pitch_obs, f_time
        else:
            return None

    def reset(self) -> None:
        pass


class PianoRollProcessor(Processor):
    """Sustained pianoroll per MIDI frame (held notes snapshot).

    Tracks ``note_on`` / ``note_off`` events across frames to maintain an
    internal "currently held" set, then emits a pianoroll vector each call
    showing which pitches are sounding right now (including notes that
    started in earlier frames and have not been released).

    Stateful: held notes persist across calls until a matching
    ``note_off`` (or a ``note_on`` with ``velocity == 0``). Empty frames
    still emit a vector reflecting whatever is held; the output is never
    ``None``.

    Parameters
    ----------
    use_velocity : bool, default False
        If True, each held pitch contributes its MIDI velocity (0-127).
        If False, contributes 1.
    piano_range : bool, default False
        If True, restrict to 88-key piano range (MIDI 21-108).
        Otherwise 128 pitches.
    dtype : type, default np.float32
        Numpy dtype of the output vector.

    Returns
    -------
    ``(pianoroll, f_time)`` where:
        - ``pianoroll``: ``np.ndarray`` of shape ``(88,)`` when
          ``piano_range=True`` else ``(128,)``. Value at pitch ``p`` is
          the velocity (if ``use_velocity``) or 1 if held, else 0.
          Example: holding C4(60) and E4(64) with default piano range
          → vector with 1 at indices 39 and 43 (=pitch-21), zeros elsewhere.
        - ``f_time``: frame timestamp from the stream.

    Notes
    -----
    Contrast with ``OnsetOnlyPianoRollProcessor``, which only encodes
    onsets (note_ons) without sustain. Use this processor when the
    downstream model needs a "what is currently sounding" snapshot
    (e.g. matching against frame-sampled audio features).
    """

    def __init__(
        self,
        use_velocity: bool = False,
        piano_range: bool = False,
        dtype: type = np.float32,
    ):
        Processor.__init__(self)
        self.active_notes: Dict = dict()
        self.piano_roll_slices: List[np.ndarray] = []
        self.use_velocity: bool = use_velocity
        self.piano_range: bool = piano_range
        self.dtype: type = dtype

    def __call__(
        self,
        frame: InputMIDIFrame,
    ) -> Tuple[np.ndarray, float]:
        # initialize piano roll
        piano_roll_slice: np.ndarray = np.zeros(128, dtype=self.dtype)
        data, f_time = frame
        for msg, m_time in data:
            if msg.type in ("note_on", "note_off"):
                if msg.type == "note_on" and msg.velocity > 0:
                    self.active_notes[msg.note] = (msg.velocity, m_time)
                else:
                    try:
                        del self.active_notes[msg.note]
                    except KeyError:
                        pass

        for note, (vel, m_time) in self.active_notes.items():
            if self.use_velocity:
                piano_roll_slice[note] = vel
            else:
                piano_roll_slice[note] = 1

        if self.piano_range:
            piano_roll_slice = piano_roll_slice[21:109]
        self.piano_roll_slices.append(piano_roll_slice)

        return piano_roll_slice, f_time

    def reset(self) -> None:
        self.piano_roll_slices = []
        self.active_notes = dict()


class OnsetOnlyPianoRollProcessor(Processor):
    """Binary pianoroll per chord onset (no sustain), with streaming chord grouping.

    Tracks only ``note_on`` events (ignores ``note_off``). Accumulates
    note-ons into a pending chord and emits the chord vector when the
    gap between consecutive ``note_on`` arrival times exceeds
    ``chord_threshold``. Notes within the gap are merged into one
    observation (chord stacking).

    Stateful: pending notes are held across calls until the next note
    arrives more than ``chord_threshold`` later. Use
    ``flush_remaining()`` at end-of-stream to emit any last pending
    chord that has no following onset.

    Parameters
    ----------
    piano_range : bool, default True
        If True, restrict to 88-key piano range (MIDI 21-108).
        Otherwise 128 pitches.
    chord_threshold : float, default ``CHORD_THRESHOLD`` (35 ms)
        Maximum gap (seconds) between consecutive ``note_on`` times to
        consider them part of the same chord. Larger gap triggers flush.
    dtype : type, default np.float32
        Numpy dtype of the output vector.

    Returns
    -------
    None while accumulating (pending chord not yet flushed) or for
    frames with no relevant ``note_on``. Otherwise a tuple
    ``(pianoroll, t)`` where:
        - ``pianoroll``: ``np.ndarray`` of shape ``(88,)`` when
          ``piano_range=True`` else ``(128,)``. Binary (0/1) with 1 at
          indices of every pitch in the flushed chord.
          Example: chord (C4, E4, G4) flushed with default piano range
          → vector with 1 at indices 39, 43, 46 (=pitch-21).
        - ``t``: time (seconds) of the FIRST ``note_on`` in the flushed
          chord — the chord onset time, not the frame emit time. Useful
          for downstream IOI computation that wants the actual chord
          onset.

    Notes
    -----
    Designed for event-level MIDI OLTW (arzt/dixon), where the score
    side is a binary onset pianoroll per unique score onset.
    """

    def __init__(
        self,
        piano_range: bool = True,
        chord_threshold: float = CHORD_THRESHOLD,
        dtype: type = np.float32,
    ):
        Processor.__init__(self)
        self.piano_range = piano_range
        self.chord_threshold = chord_threshold
        self.dtype = dtype
        self._dim = 88 if piano_range else 128
        self._offset = 21 if piano_range else 0
        self._pending_notes: List[int] = []
        self._pending_time: Optional[float] = None

    def _flush(self) -> Tuple[np.ndarray, float]:
        vec = np.zeros(self._dim, dtype=self.dtype)
        for note in self._pending_notes:
            idx = note - self._offset
            if 0 <= idx < self._dim:
                vec[idx] = 1.0
        t = self._pending_time
        self._pending_notes = []
        self._pending_time = None
        return (vec, t)

    def __call__(self, frame: InputMIDIFrame) -> Optional[Tuple[np.ndarray, float]]:
        data, _ = frame

        result = None
        for msg, m_time in data:
            if msg.type == "note_on" and msg.velocity > 0:
                # Flush previous chord if gap exceeded
                if (
                    self._pending_time is not None
                    and (m_time - self._pending_time) > self.chord_threshold
                    and self._pending_notes
                ):
                    result = self._flush()

                self._pending_notes.append(msg.note)
                if self._pending_time is None:
                    self._pending_time = m_time

        return result

    def flush_remaining(self) -> Optional[Tuple[np.ndarray, float]]:
        """Flush any remaining pending notes (call at end of stream)."""
        if self._pending_notes:
            return self._flush()
        return None

    def reset(self) -> None:
        self._pending_notes = []
        self._pending_time = None


class PitchClassPianoRollProcessor(Processor):
    """
    A class to convert a MIDI file time slice to a piano roll representation.

    Parameters
    ----------
    use_velocity : bool
        If True, the velocity of the note is used as the value in the piano
        roll. Otherwise, the value is 1.
    piano_range : bool
        If True, the piano roll will only contain the notes in the piano.
        Otherwise, the piano roll will contain all 128 MIDI notes.
    dtype : type
        The data type of the piano roll. Default is float.
    """

    def __init__(
        self,
        use_velocity: bool = False,
        dtype: type = np.float32,
    ):
        Processor.__init__(self)
        self.active_notes: Dict = dict()
        self.pitch_class_slices: List[np.ndarray] = []
        self.use_velocity: bool = use_velocity
        self.dtype: type = dtype

    def __call__(
        self,
        frame: InputMIDIFrame,
    ) -> Tuple[np.ndarray, float]:
        # initialize pitch class
        pitch_class_slice: np.ndarray = np.zeros(12, dtype=self.dtype)
        data, f_time = frame
        for msg, m_time in data:
            if msg.type in ("note_on", "note_off"):
                if msg.type == "note_on" and msg.velocity > 0:
                    self.active_notes[msg.note] = (msg.velocity, m_time)
                else:
                    try:
                        del self.active_notes[msg.note]
                    except KeyError:
                        pass

        for note, (vel, m_time) in self.active_notes.items():
            if self.use_velocity:
                pitch_class_slice[note % 12] = max(vel, pitch_class_slice[note % 12])
            else:
                pitch_class_slice[note % 12] = 1

        self.pitch_class_slices.append(pitch_class_slice)

        return pitch_class_slice, f_time

    def reset(self) -> None:
        self.pitch_class_slices = []
        self.active_notes = dict()


def compute_features_from_symbolic(
    ref_info: Union[ScoreLike, PerformanceLike, NDArray, str],
    processor_name: str,
    processor_kwargs: Optional[dict] = None,
    polling_period: Optional[float] = 0.01,
    bpm: Optional[float] = 120,
):
    processor_mapping = {
        "pitch": PitchProcessor,
        "pianoroll": PianoRollProcessor,
        "pitch_class_pianoroll": PitchClassPianoRollProcessor,
    }

    if processor_kwargs is None:
        processor_kwargs = {}

    feature_processor = processor_mapping[processor_name](**processor_kwargs)

    if isinstance(ref_info, Score):
        ref_info = performance_from_part(
            part=merge_parts(ref_info) if len(ref_info) > 1 else ref_info[0],
            bpm=bpm,
        )
    elif isinstance(ref_info, Part):
        ref_info = performance_from_part(
            part=ref_info,
            bpm=bpm,
        )
    elif isinstance(ref_info, str):
        # This method assumes that all paths are to
        # performance files.
        ref_info = pt.load_performance(ref_info)

    elif isinstance(ref_info, np.ndarray):
        ref_info = PerformedPart.from_note_array(ref_info)

    if polling_period is not None:
        frames_array, frame_times = framed_midi_messages_from_performance(
            perf=ref_info, polling_period=polling_period
        )
    else:
        frames_array, frame_times = midi_messages_from_performance(
            perf=ref_info,
        )
        # Get same format as expected by the input processors
        frames_array = np.array([list(zip(frames_array, frame_times))])

    outputs = []
    for frame, f_time in zip(frames_array, frame_times):
        output = feature_processor((frame, f_time))

        outputs.append(output)

    return outputs


# ---------------------------------------------------------------------------
# Onset-level features (for event-level OLTW)
# ---------------------------------------------------------------------------


def onset_pianoroll(
    note_array: NDArray,
    onset_key: str = "onset_beat",
    piano_range: bool = True,
) -> Tuple[NDArray, NDArray]:
    """One binary pitch vector per unique onset.

    Parameters
    ----------
    note_array : np.ndarray
        Structured array with "pitch" and ``onset_key`` fields.
    onset_key : str
        Column name for onset times ("onset_beat" or "onset_sec").
    piano_range : bool
        If True, 88 keys (MIDI 21-108). Otherwise 128.

    Returns
    -------
    features : np.ndarray [N_onsets, D]
    onsets : np.ndarray [N_onsets]
    """
    dim = 88 if piano_range else 128
    offset = 21 if piano_range else 0
    onsets = np.unique(note_array[onset_key])
    features = np.zeros((len(onsets), dim), dtype=np.float32)
    for i, t in enumerate(onsets):
        for p in note_array["pitch"][note_array[onset_key] == t]:
            idx = p - offset
            if 0 <= idx < dim:
                features[i, idx] = 1.0
    return features, onsets


def group_onsets(
    note_array: NDArray,
    threshold: float = CHORD_THRESHOLD,
    piano_range: bool = True,
) -> Tuple[NDArray, NDArray]:
    """Group note onsets within ``threshold`` seconds into chords.

    Parameters
    ----------
    note_array : np.ndarray
        Structured array with "pitch" and "onset_sec" fields.
    threshold : float
        Maximum time gap (seconds) to merge into one chord.
    piano_range : bool
        If True, 88 keys. Otherwise 128.

    Returns
    -------
    features : np.ndarray [N_chords, D]
    times : np.ndarray [N_chords]
        Mean onset time per chord group.
    """
    dim = 88 if piano_range else 128
    offset = 21 if piano_range else 0
    onsets = note_array["onset_sec"]
    si = np.argsort(onsets)
    groups, cur = [], [si[0]]
    for i in range(1, len(si)):
        if onsets[si[i]] - onsets[si[i - 1]] < threshold:
            cur.append(si[i])
        else:
            groups.append(cur)
            cur = [si[i]]
    groups.append(cur)
    features = np.zeros((len(groups), dim), dtype=np.float32)
    times = np.zeros(len(groups))
    for i, g in enumerate(groups):
        times[i] = np.mean(onsets[g])
        for idx in g:
            p = note_array["pitch"][idx] - offset
            if 0 <= p < dim:
                features[i, p] = 1.0
    return features, times


if __name__ == "__main__":  # pragma: no cover
    pass
