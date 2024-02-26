#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Input MIDI stream
"""

import os
import tempfile
import threading
import time
import numpy as np


from typing import Any, Tuple, List, Union, Optional, Callable

import mido

from mido.ports import BaseInput as MidiInputPort

from matchmaker.utils.misc import RECVQueue
from matchmaker.utils.processor import Stream, ProcessorWrapper

# Default polling period (in seconds)
POLLING_PERIOD = 0.01


class MidiStream(threading.Thread, Stream):
    """
    A class to process input MIDI stream in real time

    Parameters
    ----------
    port : mido.ports.BaseInput
        Input MIDI port

    queue : RECVQueue
        Queue to store processed MIDI input

    init_time : Optional[float]
        The initial time. If none given, the
        initial time will be set to the starting time
        of the thread.

    return_midi_messages: bool
        Return MIDI messages in addition to the
        processed features.

    mediator : Mediator or None
        A Mediator instance to filter input MIDI.
        This is useful for certain older instruments,
        like the Bösendorfer CEUS, which do not distinguish
        between notes played by a human, and notes sent
        from a different process  (e.g., an accompaniment system)
    """

    midi_in: MidiInputPort
    init_time: float
    listen: bool
    queue: RECVQueue
    features: List[Callable]
    return_midi_messages: bool
    first_message: bool

    def __init__(
        self,
        port: MidiInputPort,
        queue: RECVQueue,
        init_time: Optional[float] = None,
        features=None,
        return_midi_messages=False,
        mediator=None,
    ):
        if features is None:
            features = [ProcessorWrapper(lambda x: x)]
        threading.Thread.__init__(self)
        Stream.__init__(self, features=features)
        self.midi_in = port
        self.init_time = init_time
        self.listen = False
        self.queue = queue
        self.first_msg = False
        self.return_midi_messages = return_midi_messages
        self.mediator = mediator

    def _process_frame(self, data, *args, **kwargs) -> Tuple[np.ndarray, int]:
        """
        Parameters
        ----------
        data : MIDIFrame
        """
        self._process_feature(data)

        return (data, int(self.listen))

    def _process_feature(self, msg: mido.Message, *args, **kwargs) -> None:
        c_time = self.current_time

        # TODO: Use an OutputProcessor
        output = [proc(([(msg, c_time)], c_time))[0] for proc in self.features]
        if self.return_midi_messages:
            self.queue.put(((msg, c_time), output))
        else:
            self.queue.put(output)

    def run(self):
        self.start_listening()
        while self.listen:
            msg = self.midi_in.poll()
            if msg is not None:
                if (
                    self.mediator is not None
                    and msg.type == "note_on"
                    and self.mediator.filter_check(msg.note)
                ):
                    continue
                self._process_frame(data=msg)

    @property
    def current_time(self):
        """
        Get current time since starting to listen
        """
        return time.time() - self.init_time

    def start_listening(self):
        """
        Start listening to midi input (open input port and
        get starting time)
        """
        print("* Start listening to MIDI stream....")
        self.listen = True
        if self.init_time is None:
            self.init_time = time.time()

    def stop_listening(self):
        """
        Stop listening to MIDI input
        """
        print("* Stop listening to MIDI stream....")
        # break while loop in self.run
        self.listen = False
        # reset init time
        self.init_time = None


class Buffer(object):
    """
    A Buffer for MIDI input

    This class is a buffer to collect MIDI messages
    within a specified time window.

    Parameters
    ----------
    polling_period : float
        Polling period in seconds

    Attributes
    ----------
    polling_period : float
        Polling period in seconds.

    frame : list of tuples of (mido.Message and float)
        A list of tuples containing MIDI messages and
        the absolute time at which the messages arrived

    start : float
        The starting time of the buffer
    """

    polling_period: float
    frame: List[Tuple[mido.Message, float]]
    start: Optional[float]

    def __init__(self, polling_period: float) -> None:
        self.polling_period = polling_period
        self.frame = []
        self.start = None

    def append(self, input, time) -> None:
        self.frame.append((input, time))

    def reset(self, time) -> None:
        self.frame = []
        self.start = time

    @property
    def end(self):
        """
        Maximal end time of the frame
        """
        return self.start + self.polling_period

    @property
    def time(self):
        """
        Time of the middle of the frame
        """
        return self.start + 0.5 * self.polling_period

    def __len__(self):
        """
        Number of MIDI messages in the frame
        """
        return len(self.frame)

    def __str__(self):
        return str(self.frame)


class FramedMidiStream(MidiStream):
    def __init__(
        self,
        port,
        queue,
        polling_period=POLLING_PERIOD,
        init_time=None,
        features=None,
        return_midi_messages=False,
        mediator=None,
    ):
        MidiStream.__init__(
            port=port,
            queue=queue,
            init_time=init_time,
            features=features,
            return_midi_messages=return_midi_messages,
            mediator=mediator,
        )
        self.polling_period = polling_period

    def run(self):
        """
        TODO
        ----
        * Fix Error with c_time when stopping the thread
        * Adapt sleep time from midi_online
        """
        self.start_listening()
        frame = Buffer(self.polling_period)
        frame.start = self.current_time

        st = self.polling_period * 0.5
        while self.listen:
            time.sleep(st)
            if self.listen:
                c_time = self.current_time
                msg = self.midi_in.poll()
                if msg is not None:
                    if (
                        self.mediator is not None
                        and (msg.type == "note_on" and msg.velocity > 0)
                        and self.mediator.filter_check(msg.note)
                    ):
                        continue
                    if msg.type in ["note_on", "note_off"]:
                        frame.append(msg, self.current_time)
                        if not self.first_msg:
                            self.first_msg = True
                if c_time >= frame.end and self.first_msg:
                    output = self.pipeline((frame.frame[:], frame.time))
                    if self.return_midi_messages:
                        self.queue.put((frame.frame, output))
                    else:
                        self.queue.put(output)
                    # self.queue.put(output)
                    frame.reset(c_time)


class MockingMidiStream(MidiStream):
    """ """

    pass


class MockingFramedMidiStream(MidiStream):
    """ """

    pass
