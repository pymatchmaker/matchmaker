#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Dynamic programming-based alignment methods.

Each algorithm has a base class and two variants:
  - Frame: fixed-rate features (audio)
  - Event: onset-level features (MIDI)
"""

from .oltw_arzt import (
    OnlineTimeWarpingArzt,
    OnlineTimeWarpingArztEvent,
    OnlineTimeWarpingArztFrame,
    OnlineTimeWarpingArztTempoFrame,
)
from .oltw_dixon import (
    OnlineTimeWarpingDixon,
    OnlineTimeWarpingDixonEvent,
    OnlineTimeWarpingDixonFrame,
)
from .oltw_arzt_multi_fold import OnlineTimeWarpingArztMultiFold
