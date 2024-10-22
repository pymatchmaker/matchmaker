#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Audio device utilities.
"""
from typing import List

import pyaudio


class AudioDeviceInfo(object):
    """Info about an audio device

    Parameters
    ----------
    device_info : dict
        Info about the audio device
    device_index : int
        Index of the audio device
    """

    def __init__(
        self,
        device_info: dict,
        device_index: int,
    ) -> None:

        self.device_info = device_info
        self.device_index = device_index
        self.name = device_info["name"]
        self.input_channels = self.device_info.get("maxInputChannels", 0)
        self.output_channels = self.device_info.get("maxOutputChannels", 0)
        self.default_sample_rate = self.device_info["defaultSampleRate"]

    def __str__(self) -> str:
        out_str = (
            f"Device {self.device_index}: {self.device_info['name']}\n"
            f"  - Input Channels: {self.device_info['maxInputChannels']}\n"
            f"  - Output Channels: {self.device_info['maxOutputChannels']}\n"
            f"  - Default Sample Rate: {self.device_info['defaultSampleRate']}\n"
        )
        return out_str


def get_audio_devices() -> List[AudioDeviceInfo]:
    """Get list of audio devices

    Returns
    -------
    audio_devices : List[AudioDeviceInfo]
        List of available audio devices.
    """
    p = pyaudio.PyAudio()

    device_count = p.get_device_count()

    audio_devices = []
    for i in range(device_count):
        device_info = AudioDeviceInfo(p.get_device_info_by_index(i), i)
        audio_devices.append(device_info)
    p.terminate()
    return audio_devices


def check_input_audio_devices() -> bool:
    """
    Check whether the system has audio devices
    with input

    Returns
    -------
    has_audio_inputs : bool
        True if the system has an audio device with inputs
    """
    audio_devices = get_audio_devices()

    num_input_channels = sum([ad.input_channels for ad in audio_devices])

    has_audio_inputs = num_input_channels > 0

    return has_audio_inputs


def list_audio_devices() -> None:
    """
    Function to list all available audio devices
    """
    audio_devices = get_audio_devices()

    for ad in audio_devices:
        print(ad)


def get_device_index_from_name(device_name: str) -> int:
    """Get audio device index from name

    Parameters
    ----------
    device_name : str
        Name of th audio device.

    Returns
    -------
    int
        index of the audio device

    Raises
    ------
    ValueError
        Audio device not found.
    """
    audio_devices = get_audio_devices()

    for ad in audio_devices:
        if device_name == ad.name:
            return ad.device_index

    print(f"{device_name} not found!\n" "Here is the list of available devices:\n")
    for ad in audio_devices:
        print(ad)

    raise ValueError("Audio device `{device_name}` not found")
