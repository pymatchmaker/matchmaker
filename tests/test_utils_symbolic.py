#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Tests for matchmaker/utils/partitura.py
"""
import unittest

import matchmaker
import mido
import numpy as np
import partitura as pt
from matchmaker.utils.symbolic import (
    midi_messages_from_midi,
    framed_midi_messages_from_midi,
)

from partitura.utils.music import generate_random_performance_note_array

from tempfile import NamedTemporaryFile


class TestLoadingMethods(unittest.TestCase):
    """
    Tests for methods for loading data from symbolic files.
    """

    def test_midi_messages_from_midi(self):
        """
        Tests for `midi_messages_from_midi`.
        """
        tmp_file = NamedTemporaryFile(delete=True)
        perf = generate_random_performance_note_array(
            return_performance=True,
        )

        pt.save_performance_midi(
            perf,
            out=tmp_file.name,
        )

        filename = tmp_file.name

        midi_messages, message_times = midi_messages_from_midi(filename)

        mf = mido.MidiFile(filename)

        valid_messages = [msg for msg in mf if not isinstance(msg, mido.MetaMessage)]

        self.assertTrue(len(valid_messages) == len(midi_messages))

        self.assertTrue(np.all(np.diff(message_times) >= 0))

        tmp_file.close()

    def test_framed_midi_messages_from_midi(self):
        """
        Tests for `framed_midi_messages_from_midi`
        and indirectly `midi_messages_to_framed_midi`.
        """
        filename = matchmaker.EXAMPLE_PERFORMANCE

        polling_period = 0.01
        midi_frames, frame_times = framed_midi_messages_from_midi(
            filename,
            polling_period=polling_period,
        )

        self.assertTrue(isinstance(midi_frames, np.ndarray))
        self.assertTrue(isinstance(frame_times, np.ndarray))
        self.assertTrue(len(midi_frames) == len(frame_times))

        for msgs, ft in zip(midi_frames, frame_times):

            if len(msgs) > 0:
                for msg, t in msgs:
                    self.assertTrue(isinstance(msg, mido.Message))
                    self.assertTrue(
                        t >= ft - 0.5 * polling_period
                        and t <= ft + 0.5 * polling_period
                    )
