#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Features from symbolic files
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
from mido import Message

from utils.processor import Processor

# Type hint for Input MIDI frame. A frame is a tuple
# consisting of a list with the MIDI messages corresponding
# to the frame (List[Tuple[Message, float]]) and the
# time associated to the frame
InputMIDIFrame = Tuple[List[Tuple[Message, float]], float]


class PitchIOIProcessor(Processor):
    """
    A class to process pitch and IOI information from MIDI files

    Parameters
    ----------
    piano_range : bool
        If True, the pitch range will be limited to the piano range (21-108).
    """

    def __init__(self, piano_range: bool = False):
        super().__init__()
        self.prev_time: float = 0
        self.piano_range = piano_range

        self.pitch_bias = 21 if piano_range else 0

    def __call__(
        self,
        frame: InputMIDIFrame,
        kwargs: Dict = {},
    ) -> Tuple[Optional[np.ndarray], Dict]:
        data, f_time = frame
        pitch_obs = []

        for msg, t in data:
            if (
                getattr(msg, "type", "other") == "note_on"
                and getattr(msg, "velocity", 0) > 0
            ):
                pitch_obs.append(msg.note)

        if len(pitch_obs) > 0:
            ioi_obs = f_time - self.prev_time
            self.prev_time = f_time
            return (np.array(pitch_obs), ioi_obs), {}

        else:
            return None, {}

    def reset(self) -> None:
        pass


class PianoRollProcessor(object):
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
        dtype: type = float,
    ):
        self.active_notes: Dict = dict()
        self.piano_roll_slices: List[np.ndarray] = []
        self.use_velocity: bool = use_velocity
        self.piano_range: bool = piano_range
        self.dtype: type = dtype

    def __call__(
        self, frame: InputMIDIFrame, kwargs: Dict = {}
    ) -> Tuple[np.ndarray, Dict]:
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

        return piano_roll_slice, {}

    def reset(self) -> None:
        self.piano_roll_slices = []
        self.active_notes = dict()


class CumSumPianoRollProcessor(object):
    """
    A class to convert a MIDI file time slice to a cumulative sum piano roll
    representation.

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
        dtype: type = float,
    ) -> None:
        self.active_notes: Dict = dict()
        self.piano_roll_slices: List[np.ndarray] = []
        self.use_velocity: bool = use_velocity
        self.piano_range: bool = piano_range
        self.dtype: type = dtype

    def __call__(
        self,
        frame: InputMIDIFrame,
        kwargs: Dict = {},
    ) -> Tuple[np.ndarray, Dict]:
        # initialize piano roll
        piano_roll_slice = np.zeros(128, dtype=self.dtype)
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

        return piano_roll_slice, {}

    def reset(self) -> None:
        self.piano_roll_slices = []
        self.active_notes = dict()


if __name__ == "__main__":
    pass