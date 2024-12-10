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

from matchmaker.utils.processor import Processor
from matchmaker.utils.symbolic import (
    framed_midi_messages_from_performance,
    midi_messages_from_performance,
)
from matchmaker.utils.typing import InputMIDIFrame, NDArrayFloat


class PitchProcessor(Processor):
    """
    A class to process pitch information from MIDI input.

    Parameters
    ----------
    piano_range : bool
        If True, the pitch range will be limited to the piano range (21-108).

    return_pitch_list: bool
        If True, it will return an array of MIDI pitch values, instead of
        a "piano roll" slice.
    """

    prev_time: float
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
                return np.array(
                    pitch_obs_list,
                    dtype=np.float32,
                )
            return pitch_obs
        else:
            return None

    def reset(self) -> None:
        pass


class PitchIOIProcessor(Processor):
    """
    A class to process pitch and IOI information from MIDI files

    Parameters
    ----------
    piano_range : bool
        If True, the pitch range will be limited to the piano range (21-108).

    return_pitch_list: bool
        If True, it will return an array of MIDI pitch values, instead of
        a "piano roll" slice.
    """

    prev_time: Optional[float]
    piano_range: bool

    def __init__(
        self,
        piano_range: bool = False,
        return_pitch_list: bool = False,
    ) -> None:
        super().__init__()
        self.prev_time = None
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
            if self.prev_time is None:
                # There is no IOI for the first observed note
                ioi_obs = 0.0
            else:
                ioi_obs = f_time - self.prev_time
            self.prev_time = f_time
            if self.piano_range:
                pitch_obs = pitch_obs[21:109]

            if self.return_pitch_list:
                return (
                    np.array(
                        pitch_obs_list,
                        dtype=np.float32,
                    ),
                    ioi_obs,
                )
            return (pitch_obs, ioi_obs)
        else:
            return None

    def reset(self) -> None:
        pass


class PianoRollProcessor(Processor):
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
    ) -> np.ndarray:
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

        return piano_roll_slice

    def reset(self) -> None:
        self.piano_roll_slices = []
        self.active_notes = dict()


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
    ) -> np.ndarray:
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

        return pitch_class_slice

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
        "pitch_ioi": PitchIOIProcessor,
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


if __name__ == "__main__":  # pragma: no cover
    pass
