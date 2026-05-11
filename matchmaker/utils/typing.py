#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Type aliases used across matchmaker.

Aliases
-------
``NDArrayFloat`` / ``NDArrayInt``
    Numpy arrays with float32 / int32 dtype.

``InputMIDIFrame``
    The frame type passed from :class:`matchmaker.io.midi.MidiStream` to
    a :class:`matchmaker.features.processor.Processor`. It is a 2-tuple
    ``(messages, frame_time)``:

    - ``messages`` : ``List[Tuple[mido.Message, float]]``. Each entry is
      a MIDI message paired with the per-message arrival time
      (``m_time``). Stateful processors that group across messages
      (e.g. :class:`~matchmaker.features.midi.OnsetOnlyPianoRollProcessor`)
      use these ``m_time`` values to decide chord boundaries.
    - ``frame_time`` : ``float``. The stream's nominal time for the
      frame. Stateless processors (e.g.
      :class:`~matchmaker.features.midi.PitchProcessor`) emit this as the
      second element of their output tuple ``(features, perf_time)``.
      Stateful processors override it with their own chord-onset time.

    Examples
    ~~~~~~~~
    Single-message mode (one message per call)::

        ([(msg, 0.5)], 0.5)
         └───list────┘  └frame_time

    Windowed mode (e.g. 3 notes in a 10 ms window ``[0.500, 0.510)``)::

        ([(msg_a, 0.500), (msg_b, 0.503), (msg_c, 0.508)], 0.505)
         └─────────────────list────────────────────────┘  └frame_time

    ``frame_time`` is the window **midpoint** (``start + 0.5 * polling_period``).
    See :attr:`matchmaker.utils.symbolic.Buffer.time`.
"""

from typing import List, Tuple

import numpy as np
from mido import Message
from numpy.typing import NDArray

NDArrayFloat = NDArray[np.float32]
NDArrayInt = NDArray[np.int32]

InputMIDIFrame = Tuple[List[Tuple[Message, float]], float]
