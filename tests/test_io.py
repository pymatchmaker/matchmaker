#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the matchmaker.io module.
"""
import time
import unittest
from unittest.mock import patch

import librosa

from matchmaker.features.audio import (
    ChromagramProcessor,
    MelSpectrogramProcessor,
    MFCCProcessor,
)
from matchmaker.io.audio import MockAudioStream
from matchmaker.utils.misc import RECVQueue


class TestMockAudioStream(unittest.TestCase):
    def setUp(self):
        self.audio_file = librosa.ex("pistachio")
        sr = 22050
        hop_length = 256
        queue = RECVQueue()
        features = [ChromagramProcessor(), MFCCProcessor(), MelSpectrogramProcessor()]
        chunk_size = 1024
        self.stream = MockAudioStream(
            sample_rate=sr,
            hop_length=hop_length,
            queue=queue,
            features=features,
            chunk_size=chunk_size,
            file_path=self.audio_file,
        )

    def tearDown(self):
        self.stream.stop()

    def test_audio_stream_start(self):
        # Given: the stream is not listening
        self.assertFalse(self.stream.listen)
        self.assertFalse(self.stream.is_alive())

        # When: the stream is started
        self.stream.start()
        time.sleep(5)  # wait for stream thread to start

        # Then: the stream is listening and is alive
        self.assertTrue(self.stream.listen)
        self.assertTrue(self.stream.is_alive())

    def test_audio_stream_stop(self):
        # Given: the stream is listening and is alive
        self.stream.start()
        time.sleep(2)
        self.assertTrue(self.stream.listen)
        self.assertTrue(self.stream.is_alive())

        # When: the stream is stopped
        self.stream.stop()

        # Then: the stream is not listening but is still alive
        self.assertFalse(self.stream.listen)
        self.assertTrue(self.stream.is_alive())

    def test_run_method_called_when_started(self):
        with patch.object(self.stream, "run", wraps=self.stream.run) as mocked_run:
            self.stream.start()
            time.sleep(2)
            mocked_run.assert_called_once()

    def test_current_time(self):
        # Given: the stream is started
        self.stream.start()

        # When: the stream is stopped after 4 seconds
        start_time = time.time()
        time.sleep(4)
        self.stream.stop()

        # Then: the current time is approximately 3 seconds (with a delta of some time to start thread)
        elapsed_time = time.time() - start_time
        self.assertAlmostEqual(self.stream.current_time, elapsed_time, delta=1)

    def test_queue_not_empty_after_run(self):
        # Given: the stream is started
        self.stream.start()
        time.sleep(2)

        # Then: the queue is not empty
        self.assertFalse(self.stream.queue.empty())

    def test_queue_feature_shapes(self):
        # Given: Start the stream
        self.stream.start()
        time.sleep(2)  # wait for stream thread to start and process some data

        # Then: Check the shape of each feature in the queue
        while not self.stream.queue.empty():
            feature = self.stream.queue.get()
            features_len = (
                ChromagramProcessor().n_chroma
                + MFCCProcessor().n_mfcc
                + MelSpectrogramProcessor().n_mels
            )
            expected_shape = (
                features_len,
                int(self.stream.chunk_size / self.stream.hop_length),  # windowed frames
            )
            self.assertEqual(feature.shape, expected_shape)
