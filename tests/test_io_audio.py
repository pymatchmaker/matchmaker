#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the matchmaker.io module.
"""
import time
from typing import Optional
import unittest
from io import StringIO
from unittest.mock import patch

import librosa
import numpy as np

from matchmaker import EXAMPLE_AUDIO
from matchmaker.features.audio import (
    ChromagramProcessor,
    MelSpectrogramProcessor,
    MFCCProcessor,
)
from matchmaker.io.audio import AudioStream
from matchmaker.utils.audio import check_input_audio_devices, get_audio_devices
from matchmaker.utils.misc import RECVQueue

HAS_AUDIO_INPUT = check_input_audio_devices()

# SKIP_REASON = (not HAS_AUDIO_INPUT, "No input audio devices detected")
SKIP_REASON = (False, "No input audio devices detected")

SAMPLE_RATE = 44100
HOP_LENGTH = 256
CHUNK_SIZE = 1


class TestAudioStream(unittest.TestCase):
    def setup(
        self,
        processor_name: str = "chroma",
        file_path: Optional[str] = None,
    ):

        if processor_name == "chroma":
            processor = ChromagramProcessor(
                sample_rate=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
            )
        elif processor_name == "mfcc":
            processor = MFCCProcessor(
                sample_rate=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
            )

        elif processor_name == "mel":
            processor = MelSpectrogramProcessor(
                sample_rate=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
            )

        elif processor_name == "dummy":

            # Test default dummy processor
            processor = None

        self.stream = AudioStream(
            file_path=file_path,
            sample_rate=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            chunk_size=CHUNK_SIZE,
            processor=processor,
        )

    def teardown(self):
        self.stream.stop()

    @unittest.skipIf(*SKIP_REASON)
    @patch("sys.stdout", new_callable=StringIO)
    def test_stream_init(self, mock_stdout):
        """Test different input configurations"""
        # Test with default settings
        stream = AudioStream()

        self.assertTrue(isinstance(stream, AudioStream))

        # If a file path is set, the input device info is
        # ignored
        stream = AudioStream(
            file_path=EXAMPLE_AUDIO,
            device_name_or_index="test_device_name",
        )

        self.assertTrue(isinstance(stream, AudioStream))
        self.assertTrue(stream.input_device_index is None)

        # Test setting specific audio devices
        audio_devices = get_audio_devices()

        for ad in audio_devices:

            if ad.input_channels > 0:
                # Set audio device from name
                stream = AudioStream(
                    device_name_or_index=ad.name,
                )

                self.assertTrue(isinstance(stream, AudioStream))
                self.assertTrue(stream.input_device_index == ad.device_index)

                # Set audio device from index
                stream = AudioStream(
                    device_name_or_index=ad.device_index,
                )

                self.assertTrue(isinstance(stream, AudioStream))
                self.assertTrue(stream.input_device_index == ad.device_index)

        # Test raising error
        with self.assertRaises(ValueError):
            # raise error if a non existing device is selected
            stream = AudioStream(device_name_or_index=len(audio_devices) + 30)

    # @unittest.skipIf(*SKIP_REASON)
    @unittest.skipIf(True, "debug")
    @patch("sys.stdout", new_callable=StringIO)
    def test_live_input(self, mock_stdout):

        num_proc_frames = dict(
            chroma=0,
            mel=0,
            mfcc=0,
            dummy=0,
        )
        for processor in [
            "chroma",
            "mel",
            "mfcc",
            "dummy",
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
                    # print(processor, c_time, d_time, features.shape)
                    p_time = c_time
                    num_proc_frames[processor] += 1

                if (time.time() - init_time) >= 2:
                    crit = False

            self.stream.stop()

            self.assertTrue(features_checked)

    @unittest.skipIf(*SKIP_REASON)
    @patch("sys.stdout", new_callable=StringIO)
    def test_live_input_context_manager(self, mock_stdout):

        num_proc_frames = dict(
            chroma=0,
            mel=0,
            mfcc=0,
            dummy=0,
        )
        for processor in [
            # "chroma",
            # "mel",
            # "mfcc",
            "dummy",
        ]:

            self.setup(processor_name=processor)

            with self.stream as stream:

                init_time = time.time()
                crit = True
                # Check that we get output from the queue
                features_checked = False

                p_time = init_time
                while crit:

                    features = stream.queue.recv()
                    c_time = stream.current_time
                    if features is not None:
                        features_checked = True
                        self.assertTrue(isinstance(features, np.ndarray))
                        d_time = c_time - p_time
                        print(
                            processor,
                            c_time,
                            stream.current_time,
                            d_time,
                            features.shape,
                        )
                        p_time = c_time
                        num_proc_frames[processor] += 1

                    if stream.current_time >= 2:
                        crit = False

            self.assertTrue(features_checked)

    def test_offline_input(self):
        for processor in [
            # "chroma",
            # "mel",
            # "mfcc",
            "dummy",
        ]:
            self.setup(processor_name=processor, file_path=EXAMPLE_AUDIO,)

            with self.stream as stream:

                while True:
                    features = stream.queue.recv()

                    print(features)









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
