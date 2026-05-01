#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains aliases for typing
"""

from typing import List, Tuple

import numpy as np
from mido import Message
from numpy.typing import NDArray

# Alias for typing arrays of a specific numerical dtype
NDArrayFloat = NDArray[np.float32]
NDArrayInt = NDArray[np.int32]


# Type hint for the MIDI frame passed from MidiStream to a Processor.
# A 2-tuple (messages, frame_time):
#   - messages : list of (mido.Message, m_time) pairs. Each m_time is
#     the per-message arrival time; buffering processors
#     (PitchChordProcessor, OnsetPianoRollProcessor) use these to decide
#     chord grouping.
#   - frame_time : the stream's nominal time for the frame. Pass-through
#     processors return this as the second element of their output tuple
#     (features, perf_time); buffering processors override it with their
#     own chord onset time.
#
# Examples
# --------
# Single-message mode (one message per call):
#     ([(msg, 0.5)], 0.5)
#      └───list────┘  └frame_time
#
# Windowed mode (e.g. 3 notes in a 10ms window):
#     ([(msg_a, 0.500), (msg_b, 0.503), (msg_c, 0.508)], 0.510)
#      └─────────────────list────────────────────────┘  └frame_time
InputMIDIFrame = Tuple[List[Tuple[Message, float]], float]
