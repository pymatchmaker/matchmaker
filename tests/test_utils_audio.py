#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the utils/audio.py module.
"""

import os
import runpy
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

# Reuse the CI guard and initialization-error handling in the audio utilities.
HAS_AUDIO_DEVICES = bool(get_audio_devices())


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


class TestAudioDeviceAvailability(unittest.TestCase):
    def test_module_import_in_ci_does_not_initialize_audio(self):
        with (
            patch.dict(os.environ, {"CI": "true"}),
            patch(
                "pyaudio.PyAudio",
                side_effect=OSError(-9999, "Unanticipated host error"),
            ) as initialize_audio,
        ):
            module = runpy.run_path(__file__)
        self.assertFalse(module["HAS_AUDIO_DEVICES"])
        initialize_audio.assert_not_called()

    def test_module_import_handles_unavailable_audio_host(self):
        with (
            patch.dict(os.environ, {"CI": "", "GITHUB_ACTIONS": ""}),
            patch(
                "pyaudio.PyAudio",
                side_effect=OSError(-9999, "Unanticipated host error"),
            ) as initialize_audio,
            patch("builtins.print"),
        ):
            module = runpy.run_path(__file__)
        self.assertFalse(module["HAS_AUDIO_DEVICES"])
        initialize_audio.assert_called_once()
