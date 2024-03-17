#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Top of the module for input streams
"""

from .audio import AudioStream, MockAudioStream
from .midi import FramedMidiStream, MidiStream, MockFramedMidiStream, MockMidiStream

__all__ = [
    "AudioStream",
    "MockingAudioStream",
    "MidiStream",
    "FramedMidiStream",
    "MockMidiStream",
    "MockFramedMidiStream",
]
