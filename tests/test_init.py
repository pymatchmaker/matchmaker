#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the package initialization.
"""
import unittest

import librosa
import numpy as np
import partitura as pt
from partitura.performance import Performance
from partitura.score import Score

import matchmaker


class TestPackageInit(unittest.TestCase):
    def test_version(self):
        """
        Test that the version variable is defined
        """
        # test that a version string exists
        self.assertTrue(isinstance(matchmaker.__version__, str))

    def test_assets(self):
        """
        Test that loading the assets results in the
        expected data types
        """
        perf = pt.load_performance_midi(matchmaker.EXAMPLE_PERFORMANCE)
        score = pt.load_musicxml(matchmaker.EXAMPLE_SCORE)
        perf_from_match, alignment = pt.load_match(matchmaker.EXAMPLE_MATCH)
        audio, sr = librosa.load(matchmaker.EXAMPLE_AUDIO)

        self.assertTrue(isinstance(perf, Performance))
        self.assertTrue(isinstance(perf_from_match, Performance))
        self.assertTrue(isinstance(score, Score))
        self.assertTrue(isinstance(alignment, list))

        for al in alignment:
            self.assertTrue(isinstance(al, dict))

        self.assertTrue(isinstance(audio, np.ndarray))
        self.assertTrue(isinstance(sr, int))

        # Check that the contents of the MIDI file
        # and the match file are the same performance.
        perf_na = perf.note_array()
        perf_fm_na = perf_from_match.note_array()

        for field in ["pitch", "onset_sec", "duration_sec", "velocity"]:
            self.assertTrue(np.all(perf_na[field] == perf_fm_na[field]))
