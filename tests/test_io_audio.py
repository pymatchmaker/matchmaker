#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the matchmaker.io module.
"""
import os
import time
import unittest
from io import StringIO
from typing import Optional
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
from matchmaker.utils.processor import DummyProcessor
from tests.utils import generate_sine_wave

HAS_AUDIO_INPUT = check_input_audio_devices()

SKIP_REASON = (not HAS_AUDIO_INPUT, "No input audio devices detected")
# SKIP_REASON = (True, "No input audio devices detected")

SAMPLE_RATE = 22050
HOP_LENGTH = 256


class TestAudioStream(unittest.TestCase):

    def setup(
        self,
        processor_name: str = "dummy",
        file_path: Optional[str] = None,
        wait: bool = False,
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
            processor = DummyProcessor()

        self.stream = AudioStream(
            file_path=file_path,
            sample_rate=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            processor=processor,
            wait=wait,
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

    @unittest.skipIf(*SKIP_REASON)
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
                features, f_time = self.stream.queue.recv()

                if features is not None:
                    features_checked = True
                    self.assertTrue(isinstance(features, np.ndarray))

                    d_time = c_time - p_time
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

                    features, f_time = stream.queue.recv()
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

    @patch("sys.stdout", new_callable=StringIO)
    def test_offline_input(self, mock_stdout=None):
        print("clear queue")
        processed_frames = []
        for processor in [
            "chroma",
            # "mel",
            # "mfcc",
            "dummy",
        ]:
            file_path = generate_sine_wave()
            self.setup(
                processor_name=processor,
                file_path=file_path,
            )

            self.stream.start()
            print("current time", self.stream.current_time)
            self.stream.join()

            outputs = list(self.stream.queue.queue)

            for _, ftime in outputs:
                self.assertTrue(isinstance(ftime, float))

            processed_frames.append(len(outputs))

            os.unlink(file_path)

        processed_frames = np.array(processed_frames)

        self.assertTrue(np.all(processed_frames == processed_frames[0]))

    @unittest.skipIf(*SKIP_REASON)
    @patch("sys.stdout", new_callable=StringIO)
    def test_clear_queue(self, mock_stdout=None):

        file_path = generate_sine_wave(duration=0.2)
        processor = "dummy"
        self.setup(
            processor_name=processor,
            file_path=file_path,
            wait=True,
        )

        self.stream.start()
        self.stream.join()

        self.stream.clear_queue()
        outputs = list(self.stream.queue.queue)
        os.unlink(file_path)

        self.assertTrue(len(outputs) == 0)

    @unittest.skipIf(*SKIP_REASON)
    def test_process_frame(self):

        self.setup(
            processor_name="dummy",
        )

        # same type of input as live input
        original_data = np.zeros(HOP_LENGTH, dtype=np.float32)
        data = bytes(original_data)
        frame_count = 0
        time_info = {
            "input_buffer_adc_time": 0.0,
            "current_time": 0.0,
            "output_buffer_dac_time": 0.0,
        }
        status_flag = 0

        output, audio_continue = self.stream._process_frame(
            data=data,
            frame_count=frame_count,
            time_info=time_info,
            status_flag=status_flag,
        )

        self.assertTrue(output == data)
        self.assertTrue(isinstance(audio_continue, int))
        # There is no live audio here, so this should be 0
        self.assertTrue(audio_continue == 0)

        proc_output, f_time = self.stream.queue.recv()

        expected_output = np.concatenate(
            (np.zeros(self.stream.hop_length, dtype=np.float32), original_data)
        )
        self.assertTrue(np.all(proc_output == expected_output))
