#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the matchmaker.io module.
"""
import time
import unittest
from unittest.mock import patch
import numpy as np

import librosa

from matchmaker.features.audio import (
    ChromagramProcessor,
    MelSpectrogramProcessor,
    MFCCProcessor,
)
from matchmaker.io.audio import MockAudioStream, AudioStream
from matchmaker.utils.misc import RECVQueue
from matchmaker.utils.audio import check_input_audio_devices


HAS_AUDIO_INPUT = check_input_audio_devices()

if HAS_AUDIO_INPUT:

    class TestAudioStream(unittest.TestCase):

        def setup(self, processor_name: str = "chroma"):
            SR = 44100
            HOP_LENGTH = 256
            CHUNK_SIZE = 1

            if processor_name == "chroma":
                processor = ChromagramProcessor(
                    sample_rate=SR,
                    hop_length=HOP_LENGTH,
                )
            elif processor_name == "mfcc":
                processor = MFCCProcessor(
                    sample_rate=SR,
                    hop_length=HOP_LENGTH,
                )

            elif processor_name == "mel":
                processor = MelSpectrogramProcessor(
                    sample_rate=SR,
                    hop_length=HOP_LENGTH,
                )

            self.stream = AudioStream(
                sample_rate=SR,
                hop_length=HOP_LENGTH,
                chunk_size=CHUNK_SIZE,
                processor=processor,
            )

        def teardown(self):
            self.stream.stop()

        def test_live_input(self):

            num_proc_frames = dict(
                chroma=0,
                mel=0,
                mfcc=0,
            )
            for processor in [
                "chroma",
                "mel",
                "mfcc",
            ]:

                self.setup(processor_name=processor)
                self.stream.start()
                init_time = time.time()

                crit = True

                # Check that we get output from the queue
                features_checked = False

                p_time = init_time
                while crit:
                    c_time = time.time() - init_time
                    features = self.stream.queue.recv()

                    if features is not None:
                        features_checked = True
                        self.assertTrue(isinstance(features, np.ndarray))

                        d_time = c_time - p_time
                        print(processor, c_time, d_time, features.shape)
                        p_time = c_time
                        num_proc_frames[processor] += 1

                    if (time.time() - init_time) >= 2:
                        crit = False

                self.stream.stop()

                self.assertTrue(features_checked)

                print(num_proc_frames)


# class TestMockAudioStream(unittest.TestCase):
#     def setUp(self):
#         self.audio_file = librosa.ex("pistachio")
#         sr = 22050
#         hop_length = 256
#         queue = RECVQueue()

#         processor = ChromagramProcessor(sample_rate=sr, hop_length=hop_length)
#         # features = [
#         #     ChromagramProcessor(sample_rate=sr, hop_length=hop_length),
#         #     MFCCProcessor(sample_rate=sr, hop_length=hop_length),
#         #     MelSpectrogramProcessor(sample_rate=sr, hop_length=hop_length),
#         # ]
#         chunk_size = 1024
#         self.stream = MockAudioStream(
#             sample_rate=sr,
#             hop_length=hop_length,
#             queue=queue,
#             features=features,
#             chunk_size=chunk_size,
#             file_path=self.audio_file,
#         )

#     def tearDown(self):
#         self.stream.stop()

#     def test_audio_stream_start(self):
#         # Given: the stream is not listening
#         self.assertFalse(self.stream.listen)
#         self.assertFalse(self.stream.is_alive())

#         # When: the stream is started
#         self.stream.start()
#         time.sleep(5)  # wait for stream thread to start

#         # Then: the stream is listening and is alive
#         self.assertTrue(self.stream.listen)
#         self.assertTrue(self.stream.is_alive())

#     def test_audio_stream_stop(self):
#         # Given: the stream is listening and is alive
#         self.stream.start()
#         time.sleep(2)
#         self.assertTrue(self.stream.listen)
#         self.assertTrue(self.stream.is_alive())

#         # When: the stream is stopped
#         self.stream.stop()

#         # Then: the stream is not listening but is still alive
#         self.assertFalse(self.stream.listen)
#         self.assertTrue(self.stream.is_alive())

#     def test_run_method_called_when_started(self):
#         with patch.object(self.stream, "run", wraps=self.stream.run) as mocked_run:
#             self.stream.start()
#             time.sleep(2)
#             mocked_run.assert_called_once()

#     def test_current_time(self):
#         # Given: the stream is started
#         self.stream.start()

#         # When: the stream is stopped after 4 seconds
#         start_time = time.time()
#         time.sleep(4)
#         self.stream.stop()

#         # Then: the current time is approximately 3 seconds (with a delta of some time to start thread)
#         elapsed_time = time.time() - start_time
#         self.assertAlmostEqual(self.stream.current_time, elapsed_time, delta=1)

#     def test_queue_not_empty_after_run(self):
#         # Given: the stream is started
#         self.stream.start()
#         time.sleep(2)

#         # Then: the queue is not empty
#         self.assertFalse(self.stream.queue.empty())

#     def test_queue_feature_shapes(self):
#         # Given: Start the stream
#         self.stream.start()
#         time.sleep(2)  # wait for stream thread to start and process some data

#         # Then: Check the shape of each feature in the queue
#         while not self.stream.queue.empty():
#             feature = self.stream.queue.get()
#             features_len = (
#                 ChromagramProcessor().n_chroma
#                 + MFCCProcessor().n_mfcc
#                 + MelSpectrogramProcessor().n_mels
#             )

#             # TODO: Check the correct behavior
#             expected_shape = (
#                 int(self.stream.chunk_size / self.stream.hop_length),  # windowed frames
#                 features_len,
#             )
#             self.assertEqual(feature.shape, expected_shape)
