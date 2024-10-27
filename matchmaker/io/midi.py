#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Input MIDI stream
"""

import time
from types import TracebackType
from typing import Callable, List, Optional, Tuple, Type, Union

import mido
import partitura as pt
from mido.ports import BaseInput as MidiInputPort
from partitura.performance import Performance, PerformanceLike, PerformedPart

from matchmaker.io.mediator import CeusMediator
from matchmaker.utils.misc import RECVQueue
from matchmaker.utils.processor import DummyProcessor, Processor
from matchmaker.utils.stream import Stream
from matchmaker.utils.symbolic import (
    framed_midi_messages_from_performance,
    midi_messages_from_performance,
    Buffer,
)

# Default polling period (in seconds)
POLLING_PERIOD = 0.01


class MidiStream(Stream):
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
    processor: Callable
    return_midi_messages: bool
    first_message: bool
    mediator: CeusMediator
    is_windowed: bool
    # mock_stream: bool
    polling_period: Optional[float]

    def __init__(
        self,
        processor: Optional[Union[Callable, Processor]] = None,
        file_path: Optional[str] = None,
        polling_period: Optional[float] = POLLING_PERIOD,
        port: Optional[MidiInputPort] = None,
        queue: RECVQueue = None,
        init_time: Optional[float] = None,
        return_midi_messages: bool = False,
        mediator: Optional[CeusMediator] = None,
    ):
        if processor is None:
            processor = DummyProcessor()

        Stream.__init__(
            self,
            processor=processor,
            mock=file_path is not None,
        )

        if file_path is not None:
            # Do not open a MIDI port for running
            # stream offline
            port = None
        self.file_path = file_path
        self.midi_in = port
        self.init_time = init_time
        self.listen = False
        self.queue = queue or RECVQueue()
        self.first_msg = False
        self.return_midi_messages = return_midi_messages
        self.mediator = mediator

        self.polling_period = polling_period
        if (polling_period is None) and (self.mock is False):
            self.is_windowed = False
            self.run = self.run_online_single
            self._process_frame = self._process_frame_message

        elif (polling_period is None) and (self.mock is True):
            self.is_windowed = False
            self.run = self.run_offline_single
            self._process_frame = self._process_frame_message

        elif (polling_period is not None) and (self.mock is False):
            self.is_windowed = True
            self.run = self.run_online_windowed
            self._process_frame = self._process_frame_window

        elif (polling_period is not None) and (self.mock is True):
            self.is_windowed = True
            self.run = self.run_offline_windowed
            self._process_frame = self._process_frame_window

    def _process_frame_message(
        self,
        data: mido.Message,
        *args,
        c_time: float,
        **kwargs,
    ) -> None:

        # output = [proc(([(msg, c_time)], c_time))[0] for proc in self.processor]
        output = self.processor(([(data, c_time)], c_time))
        if self.return_midi_messages:
            self.queue.put(((data, c_time), output))
        else:
            self.queue.put(output)

    def _process_frame_window(
        self,
        data: Buffer,
        *args,
        **kwargs,
    ) -> None:
        # the data is the Buffer instance
        output = self.processor((data.frame[:], data.time))
        # output = self.pipeline((frame.frame[:], frame.time))
        if self.return_midi_messages:
            self.queue.put((data.frame, output))
        else:
            self.queue.put(output)

    def run_online_single(self):
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
                self._process_frame_message(
                    data=msg,
                    c_time=self.current_time,
                )

    def run_online_windowed(self):
        """ """
        self.start_listening()
        frame = Buffer(self.polling_period)
        frame.start = self.current_time

        # TODO: check the effect of smaller st
        st = self.polling_period * 0.001
        while self.listen:
            time.sleep(st)

            if self.listen:
                # added if to check once again after sleep
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
                    self._process_frame_window(data=frame)
                    frame.reset(c_time)

    def run_offline_single(self):
        """
        Simulate real-time stream as loop iterating
        over MIDI messages
        """
        midi_messages, message_times = midi_messages_from_performance(
            perf=self.file_path,
        )
        self.init_time = message_times.min()
        self.start_listening()
        for msg, c_time in zip(midi_messages, message_times):
            self._process_frame_message(
                data=msg,
                c_time=c_time,
            )
        self.stop_listening()

    def run_offline_windowed(self):
        """
        Simulate real-time stream as loop iterating
        over MIDI messages
        """
        midi_frames, frame_times = framed_midi_messages_from_performance(
            perf=self.file_path,
            polling_period=self.polling_period,
        )
        self.init_time = frame_times.min()
        for frame, f_time in zip(midi_frames, frame_times):
            self._process_frame_window(
                data=frame,
            )

    @property
    def current_time(self) -> Optional[float]:
        """
        Get current time since starting to listen
        """
        if self.init_time is None:
            # TODO: Check if this has weird consequences
            self.init_time = time.time()
            return 0

        return time.time() - self.init_time

        # return time.time() - self.init_time if self.init_time is not None else None

    def start_listening(self):
        """
        Start listening to midi input (open input port and
        get starting time)
        """
        print("* Start listening to MIDI stream....")
        self.listen = True
        # set initial time
        self.current_time
        # if self.init_time is None:
        #     self.init_time = time.time()

    def stop_listening(self):
        """
        Stop listening to MIDI input
        """
        print("* Stop listening to MIDI stream....")
        # break while loop in self.run
        self.listen = False
        # reset init time
        self.init_time = None

    def __enter__(self) -> None:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        self.stop()
        if exc_type is not None:  # pragma: no cover
            # Returning True will suppress the exception
            # False means the exception will propagate
            return False
        return True

    def stop(self):
        self.stop_listening()
        self.join()

    def clear_queue(self):
        if self.queue.not_empty:
            self.queue.queue.clear()


# class FramedMidiStream(MidiStream):
#     """
#     A class to process input MIDI stream in real time into frames.
#     The main difference with MidiStream is that this class will
#     produce an output every frame, rather than every MIDI message.

#     This class is better for e.g., piano-roll like features.

#     Parameters
#     ----------
#     port : mido.ports.BaseInput
#         Input MIDI port

#     queue : RECVQueue
#         Queue to store processed MIDI input

#     polling_period : float
#         Size of the frame. This is equivalent to hop size for
#         audio processing.

#     init_time : Optional[float]
#         The initial time. If none given, the
#         initial time will be set to the starting time
#         of the thread.

#     return_midi_messages: bool
#         Return MIDI messages in addition to the
#         processed features.

#     mediator : CeusMediator or None
#         A Mediator instance to filter input MIDI.
#         This is useful for certain older instruments,
#         like the Bösendorfer CEUS, which do not distinguish
#         between notes played by a human, and notes sent
#         from a different process  (e.g., an accompaniment system)
#     """

#     def __init__(
#         self,
#         port: MidiInputPort,
#         queue: RECVQueue = None,
#         polling_period: float = POLLING_PERIOD,
#         init_time: Optional[float] = None,
#         features: Optional[List[Callable]] = None,
#         return_midi_messages: bool = False,
#         mediator: Optional[CeusMediator] = None,
#     ):
#         MidiStream.__init__(
#             self,
#             port=port,
#             queue=queue,
#             init_time=init_time,
#             processor=features,
#             return_midi_messages=return_midi_messages,
#             mediator=mediator,
#         )
#         self.polling_period = polling_period

#     def _process_feature(
#         self,
#         data: Buffer,
#         *args,
#         **kwargs,
#     ) -> None:
#         # the data is the Buffer instance
#         output = [proc((data.frame[:], data.time))[0] for proc in self.processor]
#         # output = self.pipeline((frame.frame[:], frame.time))
#         if self.return_midi_messages:
#             self.queue.put((data.frame, output))
#         else:
#             self.queue.put(output)

#     def run(self):
#         """ """
#         self.start_listening()
#         frame = Buffer(self.polling_period)
#         frame.start = self.current_time

#         # TODO: check the effect of smaller st
#         # st = self.polling_period * 0.01
#         while self.listen:
#             # time.sleep(st)
#             if self.listen:
#                 c_time = self.current_time
#                 msg = self.midi_in.poll()
#                 if msg is not None:
#                     if (
#                         self.mediator is not None
#                         and (msg.type == "note_on" and msg.velocity > 0)
#                         and self.mediator.filter_check(msg.note)
#                     ):
#                         continue
#                     if msg.type in ["note_on", "note_off"]:
#                         frame.append(msg, self.current_time)
#                         if not self.first_msg:
#                             self.first_msg = True
#                 if c_time >= frame.end and self.first_msg:

#                     self._process_feature(data=frame)
#                     frame.reset(c_time)


# class MockMidiStream(MidiStream):
#     """
#     A class to process a MIDI file offline,
#     simulating the behavior of MidiStream.
#     This class is useful for testing and evaluation.
#     """

#     file_path: Optional[str]
#     perf_data: PerformanceLike

#     def __init__(
#         self,
#         file_path: str,
#         queue: RECVQueue,
#         features: Optional[List[Callable]] = None,
#         return_midi_messages: bool = False,
#         mediator: Optional[CeusMediator] = None,
#     ):
#         MidiStream.__init__(
#             self,
#             port=None,
#             queue=queue,
#             init_time=None,
#             processor=features,
#             return_midi_messages=return_midi_messages,
#             mediator=mediator,
#         )
#         if isinstance(file_path, (Performance, PerformedPart)):
#             self.perf_data = file_path
#             self.file_path = None
#         elif isinstance(file_path, str):
#             self.perf_data = pt.load_performance(file_path)
#             self.file_path = file_path
#         else:
#             raise ValueError(
#                 "`file_path` is expected to be a string or a "
#                 "`partitura.performance.PerformanceLike` object, "
#                 f"but is {type(file_path)}"
#             )

#     def mock_stream(self):
#         """
#         Simulate real-time stream as loop iterating
#         over MIDI messages
#         """
#         midi_messages, message_times = midi_messages_from_performance(
#             perf=self.perf_data,
#         )
#         self.init_time = message_times.min()
#         self.start_listening()
#         for msg, c_time in zip(midi_messages, message_times):
#             self._process_feature(
#                 msg=msg,
#                 c_time=c_time,
#             )
#         self.stop_listening()

#     def run(self):
#         print(f"* [Mocking] Loading existing MIDI file({self.file_path})....")
#         self.mock_stream()


# class MockFramedMidiStream(FramedMidiStream):
#     """
#     A class to process a MIDI file offline,
#     simulating the behavior of FramedMidiStream.
#     This class is useful for testing and evaluation.
#     """

#     file_path: Optional[str]
#     perf_data: PerformanceLike

#     def __init__(
#         self,
#         file_path: Union[str, PerformanceLike],
#         queue: RECVQueue = None,
#         polling_period: float = POLLING_PERIOD,
#         features: Optional[List[Callable]] = None,
#         return_midi_messages: bool = False,
#         mediator: Optional[CeusMediator] = None,
#     ):
#         FramedMidiStream.__init__(
#             self,
#             port=None,
#             queue=queue,
#             polling_period=polling_period,
#             init_time=None,
#             features=features,
#             return_midi_messages=return_midi_messages,
#             mediator=mediator,
#         )

#         if isinstance(file_path, (Performance, PerformedPart)):
#             self.perf_data = file_path
#             self.file_path = None
#         elif isinstance(file_path, str):
#             self.perf_data = pt.load_performance(file_path)
#             self.file_path = file_path
#         else:
#             raise ValueError(
#                 "`file_path` is expected to be a string or a "
#                 "`partitura.performance.PerformanceLike` object, "
#                 f"but is {type(file_path)}"
#             )

#     def _process_feature(
#         self,
#         frame: List[Tuple[mido.Message, float]],
#         f_time: float,
#         *args,
#         **kwargs,
#     ) -> None:
#         # the data is the Buffer instance
#         output = [proc((frame, f_time))[0] for proc in self.processor]
#         if self.return_midi_messages:
#             self.queue.put((frame, output))
#         else:
#             self.queue.put(output)

#     def mock_stream(self):
#         """
#         Simulate real-time stream as loop iterating
#         over MIDI messages
#         """
#         midi_frames, frame_times = framed_midi_messages_from_performance(
#             perf=self.perf_data,
#             polling_period=self.polling_period,
#         )
#         self.init_time = frame_times.min()
#         for frame, f_time in zip(midi_frames, frame_times):
#             self._process_feature(
#                 frame=frame,
#                 f_time=f_time,
#             )

#     def run(self):
#         print(f"* [Mocking] Loading existing MIDI file({self.file_path})....")
#         self.start_listening()
#         self.mock_stream()
#         self.stop_listening()
