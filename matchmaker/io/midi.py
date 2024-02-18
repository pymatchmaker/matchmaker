#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Input MIDI stream
"""

import os
import tempfile
import threading
import time


from typing import Any, Tuple, List, Union, Optional, Callable

import mido

from mido.ports import BaseInput as MidiInputPort

from matchmaker.utils.misc import RECVQueue
from matchmaker.utils.processor import Stream, DummySequentialOutputProcessor

# Default polling period (in seconds)
POLLING_PERIOD = 0.01


class MidiStream(threading.Thread, Stream):
    """
    Input MIDI stream
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
            features = DummySequentialOutputProcessor()
        threading.Thread.__init__(self)
        Stream.__init__(features=features)
        self.midi_in = port
        self.init_time = init_time
        self.listen = False
        self.queue = queue
        self.first_msg = False
        self.return_midi_messages = return_midi_messages
        self.mediator = mediator

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

                c_time = self.current_time
                # To have the same output as other MidiThreads
                output = self.pipeline([(msg, c_time)], c_time)
                if self.return_midi_messages:
                    self.queue.put(((msg, c_time), output))
                else:
                    self.queue.put(output)

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

        self.listen = True
        if self.init_time is None:
            self.init_time = time.time()

    def stop_listening(self):
        """
        Stop listening to MIDI input
        """
        # break while loop in self.run
        self.listen = False
        # reset init time
        self.init_time = None

        # self.terminate()

        # Join thread
        # self.join()


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


class FramedMidiInputThread:
    def __init__(
        self,
        port,
        queue,
        polling_period=POLLING_PERIOD,
        init_time=None,
        pipeline=None,
        return_midi_messages=False,
        mediator=None,
    ):
        super().__init__(
            port=port,
            queue=queue,
            init_time=init_time,
            pipeline=pipeline,
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
        # TODO: Adapt from midi_online to allow for variable polling
        # periods?
        st = self.polling_period * 0.5
        while self.listen:
            time.sleep(st)
            if self.listen:
                # added if to check once again after sleep
                # TODO verify if still correct
                c_time = self.current_time
                msg = self.midi_in.poll()
                if msg is not None:
                    # print("Received msg:", msg)
                    if (
                        self.mediator is not None
                        and (msg.type == "note_on" and msg.velocity > 0)
                        and self.mediator.filter_check(msg.note)
                    ):
                        # print('filtered', msg)
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


def create_midi_stream(
    port,
    polling_period,
    pipeline,
    return_midi_messages=False,
    thread=False,
    mediator=None,
):
    """
    Helper to create a FramedMidiInputProcess and its respective pipe.
    """

    if thread:
        p_output = None
        p_input = RECVQueue()
        mt = FramedMidiInputThread(
            port=port,
            queue=p_input,
            polling_period=polling_period,
            pipeline=pipeline,
            return_midi_messages=return_midi_messages,
            mediator=mediator,
        )
    else:

        p_output, p_input = Pipe()
        mt = FramedMidiInputProcess(
            port=port,
            pipe=p_output,
            polling_period=polling_period,
            pipeline=pipeline,
            return_midi_messages=return_midi_messages,
            mediator=mediator,
        )

    return p_output, p_input, mt
