#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Input MIDI stream
"""

import os
import tempfile
import threading
import time
from typing import Any, Callable, List, Optional, Tuple, Union

import mido
import numpy as np
import partitura as pt
from mido.ports import BaseInput as MidiInputPort
from partitura.performance import Performance, PerformanceLike, PerformedPart

from matchmaker.io.mediator import CeusMediator
from matchmaker.utils.misc import RECVQueue
from matchmaker.utils.processor import ProcessorWrapper, Stream
from matchmaker.utils.symbolic import (
    framed_midi_messages_from_performance,
    midi_messages_from_performance,
)

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

    mediator : CeusMediator or None
        A Mediator instance to filter input MIDI.
        This is useful for certain older instruments,
        like the Bösendorfer CEUS, which do not distinguish
        between notes played by a human, and notes sent
        from a different process  (e.g., an accompaniment system)
    """

    midi_in: Optional[MidiInputPort]
    init_time: float
    listen: bool
    queue: RECVQueue
    features: List[Callable]
    return_midi_messages: bool
    first_message: bool
    mediator: CeusMediator

    def __init__(
        self,
        port: MidiInputPort,
        queue: RECVQueue,
        init_time: Optional[float] = None,
        features: Optional[List[Callable]] = None,
        return_midi_messages: bool = False,
        mediator: Optional[CeusMediator] = None,
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

    def _process_frame(
        self, data: mido.Message, c_time: float, *args, **kwargs
    ) -> Tuple[Any, int]:
        """
        Parameters
        ----------
        data : mido.Message
            Input data to the frame
        """
        self._process_feature(msg=data, c_time=c_time, *args, **kwargs)

        return (data, int(self.listen))

    def _process_feature(
        self, msg: mido.Message, *args, c_time: float, **kwargs
    ) -> None:

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
                self._process_frame(
                    data=msg,
                    c_time=self.current_time,
                )

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
    """
    A class to process input MIDI stream in real time into frames.
    The main difference with MidiStream is that this class will
    produce an output every frame, rather than every MIDI message.

    This class is better for e.g., piano-roll like features.

    Parameters
    ----------
    port : mido.ports.BaseInput
        Input MIDI port

    queue : RECVQueue
        Queue to store processed MIDI input

    polling_period : float
        Size of the frame. This is equivalent to hop size for
        audio processing.

    init_time : Optional[float]
        The initial time. If none given, the
        initial time will be set to the starting time
        of the thread.

    return_midi_messages: bool
        Return MIDI messages in addition to the
        processed features.

    mediator : CeusMediator or None
        A Mediator instance to filter input MIDI.
        This is useful for certain older instruments,
        like the Bösendorfer CEUS, which do not distinguish
        between notes played by a human, and notes sent
        from a different process  (e.g., an accompaniment system)
    """

    def __init__(
        self,
        port: MidiInputPort,
        queue: RECVQueue,
        polling_period: float = POLLING_PERIOD,
        init_time: Optional[float] = None,
        features: Optional[List[Callable]] = None,
        return_midi_messages: bool = False,
        mediator: Optional[CeusMediator] = None,
    ):
        MidiStream.__init__(
            self,
            port=port,
            queue=queue,
            init_time=init_time,
            features=features,
            return_midi_messages=return_midi_messages,
            mediator=mediator,
        )
        self.polling_period = polling_period

    def _process_feature(
        self,
        data: Buffer,
        *args,
        **kwargs,
    ) -> None:
        # the data is the Buffer instance
        output = [proc((data.frame[:], data.time))[0] for proc in self.features]
        # output = self.pipeline((frame.frame[:], frame.time))
        if self.return_midi_messages:
            self.queue.put((data.frame, output))
        else:
            self.queue.put(output)

    def run(self):
        """ """
        self.start_listening()
        frame = Buffer(self.polling_period)
        frame.start = self.current_time

        # TODO: check the effect of smaller st
        # st = self.polling_period * 0.01
        while self.listen:
            # time.sleep(st)
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

                    self._process_feature(data=frame)
                    frame.reset(c_time)


class MockMidiStream(MidiStream):
    """
    A class to process a MIDI file offline,
    simulating the behavior of MidiStream.
    This class is useful for testing and evaluation.
    """

    file_path: Optional[str]
    perf_data: PerformanceLike

    def __init__(
        self,
        file_path: str,
        queue: RECVQueue,
        features: Optional[List[Callable]] = None,
        return_midi_messages: bool = False,
        mediator: Optional[CeusMediator] = None,
    ):
        MidiStream.__init__(
            self,
            port=None,
            queue=queue,
            init_time=None,
            features=features,
            return_midi_messages=return_midi_messages,
            mediator=mediator,
        )
        if isinstance(file_path, (Performance, PerformedPart)):
            self.perf_data = file_path
            self.file_path = None
        elif isinstance(file_path, str):
            self.perf_data = pt.load_performance(file_path)
            self.file_path = file_path
        else:
            raise ValueError(
                "`file_path` is expected to be a string or a "
                "`partitura.performance.PerformanceLike` object, "
                f"but is {type(file_path)}"
            )

    def mock_stream(self):
        """
        Simulate real-time stream as loop iterating
        over MIDI messages
        """
        midi_messages, message_times = midi_messages_from_performance(
            perf=self.perf_data,
        )
        self.init_time = message_times.min()
        self.start_listening()
        for msg, c_time in zip(midi_messages, message_times):
            self._process_feature(
                msg=msg,
                c_time=c_time,
            )
        self.stop_listening()

    def run(self):
        print(f"* [Mocking] Loading existing MIDI file({self.file_path})....")
        self.mock_stream()


class MockFramedMidiStream(FramedMidiStream):
    """"""

    file_path: Optional[str]
    perf_data: PerformanceLike

    def __init__(
        self,
        file_path: Union[str, PerformanceLike],
        queue: RECVQueue,
        polling_period: float = POLLING_PERIOD,
        features: Optional[List[Callable]] = None,
        return_midi_messages: bool = False,
        mediator: Optional[CeusMediator] = None,
    ):
        FramedMidiStream.__init__(
            self,
            port=None,
            queue=queue,
            polling_period=polling_period,
            init_time=None,
            features=features,
            return_midi_messages=return_midi_messages,
            mediator=mediator,
        )

        if isinstance(file_path, (Performance, PerformedPart)):
            self.perf_data = file_path
            self.file_path = None
        elif isinstance(file_path, str):
            self.perf_data = pt.load_performance(file_path)
            self.file_path = file_path
        else:
            raise ValueError(
                "`file_path` is expected to be a string or a "
                "`partitura.performance.PerformanceLike` object, "
                f"but is {type(file_path)}"
            )

    def _process_feature(
        self,
        frame: List[Tuple[mido.Message, float]],
        f_time: float,
        *args,
        **kwargs,
    ) -> None:
        # the data is the Buffer instance
        output = [proc((frame, f_time))[0] for proc in self.features]
        if self.return_midi_messages:
            self.queue.put((frame, output))
        else:
            self.queue.put(output)

    def mock_stream(self):
        """
        Simulate real-time stream as loop iterating
        over MIDI messages
        """
        midi_frames, frame_times = framed_midi_messages_from_performance(
            perf=self.perf_data,
            polling_period=self.polling_period,
        )
        self.init_time = frame_times.min()
        for frame, f_time in zip(midi_frames, frame_times):
            self._process_feature(
                frame=frame,
                f_time=f_time,
            )

    def run(self):
        print(f"* [Mocking] Loading existing MIDI file({self.file_path})....")
        self.start_listening()
        self.mock_stream()
        self.stop_listening()
