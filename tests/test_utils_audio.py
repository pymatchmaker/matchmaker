#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the utils/audio.py module.
"""
import unittest
from io import StringIO
from unittest.mock import patch

import pyaudio

from matchmaker.utils.audio import (
    AudioDeviceInfo,
    check_input_audio_devices,
    get_audio_devices,
    get_device_index_from_name,
    list_audio_devices,
)


def has_audio_devices():
    """Check if system has audio devices"""
    p = pyaudio.PyAudio()

    DEVICE_COUNT = p.get_device_count()

    has_audio = DEVICE_COUNT > 0

    return has_audio


HAS_AUDIO_DEVICES = has_audio_devices()


SKIP_REASON = (not HAS_AUDIO_DEVICES, "No audio device detected")


class TestUtilsAudio(unittest.TestCase):
    @unittest.skipIf(*SKIP_REASON)
    def test_AudioDeviceInfo(self) -> None:

        p = pyaudio.PyAudio()

        device_info = AudioDeviceInfo(
            device_info=p.get_device_info_by_index(0), device_index=0
        )

        out_str = device_info.__str__()

        self.assertTrue(isinstance(out_str, str))

    @unittest.skipIf(*SKIP_REASON)
    def test_get_audio_devices(self) -> None:

        audio_devices = get_audio_devices()

        self.assertTrue(isinstance(audio_devices, list))

        for ad in audio_devices:
            self.assertTrue(isinstance(ad, AudioDeviceInfo))

    @unittest.skipIf(*SKIP_REASON)
    def test_check_input_audio_devices(self) -> None:

        has_audio_inputs = check_input_audio_devices()

        self.assertTrue(isinstance(has_audio_inputs, bool))

    @unittest.skipIf(*SKIP_REASON)
    @patch(
        "matchmaker.utils.audio.get_audio_devices",
        return_value=["Device 1", "Device 2"],
    )
    @patch("sys.stdout", new_callable=StringIO)
    def test_list_audio_devices_runs(self, mock_stdout, mock_get_audio_devices):
        list_audio_devices()

        # Check that the output was printed to stdout
        output = mock_stdout.getvalue()

        # Assert that the function printed the mocked audio devices
        self.assertIn("Device 1", output)
        self.assertIn("Device 2", output)

        # Assert that the mock was called (meaning the function ran)
        mock_get_audio_devices.assert_called_once()

    @unittest.skipIf(*SKIP_REASON)
    @patch("sys.stdout", new_callable=StringIO)
    def test_get_device_index_from_name(self, mock_stdout) -> None:

        # Test existing audio devices
        audio_devices = get_audio_devices()

        for ad in audio_devices:

            index = get_device_index_from_name(ad.name)
            self.assertTrue(isinstance(index, int))

        # Test raising error
        with self.assertRaises(ValueError):
            get_device_index_from_name("NoT a ReAl AuDiO dEvIcE")
