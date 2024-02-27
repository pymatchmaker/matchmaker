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
from matchmaker.utils.symbolic import midi_messages_from_midi


class TestLoadingMethods(unittest.TestCase):

    def test_midi_messages_from_midi(self):

        filename = matchmaker.EXAMPLE_PERFORMANCE

        midi_messages, message_times = midi_messages_from_midi(filename)

        mf = mido.MidiFile(filename)

        valid_messages = [msg for msg in mf if not isinstance(msg, mido.MetaMessage)]

        self.assertTrue(len(valid_messages) == len(midi_messages))

        self.assertTrue(np.all(np.diff(message_times) >= 0))
            

