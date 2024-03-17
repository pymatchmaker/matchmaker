#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains aliases for typing
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from mido import Message


# Alias for typing arrays of a specific numerical dtype
NDArrayFloat = NDArray[np.float32]
NDArrayInt = NDArray[np.int32]


# Type hint for Input MIDI frame. A frame is a tuple
# consisting of a list with the MIDI messages corresponding
# to the frame (List[Tuple[Message, float]]) and the
# time associated to the frame
InputMIDIFrame = Tuple[List[Tuple[Message, float]], float]