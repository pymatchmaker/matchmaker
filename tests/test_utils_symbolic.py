#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Tests for matchmaker/utils/partitura.py
"""
import time
import unittest
from tempfile import NamedTemporaryFile

import mido
import numpy as np
import partitura as pt
from partitura.utils.music import generate_random_performance_note_array

import matchmaker
from matchmaker.utils.symbolic import (
    Buffer,
    framed_midi_messages_from_midi,
    midi_messages_from_midi,
    midi_messages_from_performance,
)


class TestBuffer(unittest.TestCase):

    def setUp(self):
        self.polling_period = 1.0  # 1-second polling period
        self.buffer = Buffer(self.polling_period)

    def test_initialization(self):
        self.assertEqual(self.buffer.polling_period, self.polling_period)
        self.assertEqual(self.buffer.frame, [])
        self.assertIsNone(self.buffer.start)

    def test_append(self):
        msg = mido.Message("note_on", note=60)
        current_time = time.time()
        self.buffer.append(msg, current_time)
        self.assertEqual(len(self.buffer), 1)
        self.assertEqual(self.buffer.frame[0], (msg, current_time))

    def test_reset(self):
        start_time = time.time()
        self.buffer.reset(start_time)
        self.assertEqual(self.buffer.frame, [])
        self.assertEqual(self.buffer.start, start_time)

    def test_end_property(self):
        start_time = time.time()
        self.buffer.reset(start_time)
        expected_end = start_time + self.polling_period
        self.assertEqual(self.buffer.end, expected_end)

    def test_time_property(self):
        start_time = time.time()
        self.buffer.reset(start_time)
        expected_time = start_time + 0.5 * self.polling_period
        self.assertEqual(self.buffer.time, expected_time)

    def test_len(self):
        self.assertEqual(len(self.buffer), 0)
        msg = mido.Message("note_on", note=60)
        current_time = time.time()
        self.buffer.append(msg, current_time)
        self.assertEqual(len(self.buffer), 1)

    def test_str(self):
        msg = mido.Message("note_on", note=60)
        current_time = time.time()
        self.buffer.append(msg, current_time)
        expected_str = str(self.buffer.frame)
        self.assertEqual(str(self.buffer), expected_str)


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

        for buffer, ft in zip(midi_frames, frame_times):

            self.assertTrue(isinstance(buffer, Buffer))
            if len(buffer) > 0:
                for msg, t in buffer.frame:
                    self.assertTrue(isinstance(msg, mido.Message))
                    self.assertTrue(
                        t >= ft - 0.5 * polling_period
                        and t <= ft + 0.5 * polling_period
                    )

    def test_midi_messages_from_performance(self):

        filename = matchmaker.EXAMPLE_PERFORMANCE
        performance = pt.load_performance_midi(filename)
        performed_part = performance[0]
        note_array = performance.note_array()

        msgs_filename, msgs_times_filename = midi_messages_from_performance(filename)
        msgs_performance, msgs_times_performance = midi_messages_from_performance(
            performance
        )
        msgs_performed_part, msgs_times_performed_part = midi_messages_from_performance(
            performed_part
        )
        msgs_note_array, msgs_times_note_array = midi_messages_from_performance(
            note_array
        )

        # Test times (not from note_array, since the message times
        # also include pedals and controls, which are not included
        # in note arrays)
        self.assertTrue(np.all(msgs_times_filename == msgs_times_performance))
        self.assertTrue(np.all(msgs_times_filename == msgs_times_performed_part))

        self.assertTrue(all(msgs_filename == msgs_performance))
        self.assertTrue(all(msgs_performed_part == msgs_performance))

        note_on_messages = [
            msg
            for msg in msgs_filename
            if msg.type == "note_on" or msg.type == "note_off"
        ]

        self.assertTrue(len(msgs_note_array) == len(note_on_messages))

        self.assertTrue(len(msgs_times_note_array) == len(note_on_messages))
