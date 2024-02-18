#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the matchmaker.io module.
"""
import time
import unittest

import librosa

from matchmaker.features.audio import ChromagramProcessor, MFCCProcessor
from matchmaker.io.audio import MockAudioStream
from matchmaker.utils.misc import RECVQueue


class TestMockAudioStream(unittest.TestCase):
    def setUp(self):
        self.audio_file = librosa.ex("pistachio")
        y, sr = librosa.load(self.audio_file, sr=None)
        hop_length = 512
        queue = RECVQueue()
        features = [ChromagramProcessor(), MFCCProcessor()]
        chunk_size = 1024
        self.stream = MockAudioStream(
            sample_rate=sr,
            hop_length=hop_length,
            queue=queue,
            features=features,
            chunk_size=chunk_size,
            file_path=self.audio_file,
        )

    def test_is_active(self):
        self.assertFalse(self.stream.is_active)
        self.stream.run()
        self.assertTrue(self.stream.is_active)
        self.stream.stop()
        self.assertFalse(self.stream.is_active)

    def test_sample_rate(self):
        y, sr = librosa.load(self.audio_file, sr=None)
        self.assertEqual(self.stream.sample_rate, sr)

    def test_queue_not_empty_after_run(self):
        self.stream.run()
        time.sleep(1)  # wait for 1 second to allow the stream to process some data
        self.assertFalse(self.stream.queue.empty())
        self.stream.stop()
