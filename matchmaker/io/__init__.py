#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Top of the module for input streams
"""

from .audio import AudioStream
from .midi import MidiStream

__all__ = [
    "AudioStream",
    "MidiStream",
]
