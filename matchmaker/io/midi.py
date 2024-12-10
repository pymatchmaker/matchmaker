#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Input MIDI stream
"""

import time
from types import TracebackType
from typing import Callable, List, Optional, Tuple, Type, Union

import mido
from mido.ports import BaseInput as MidiInputPort

from matchmaker.features.midi import PitchIOIProcessor
from matchmaker.io.mediator import CeusMediator
from matchmaker.utils.misc import RECVQueue
from matchmaker.utils.processor import Processor
from matchmaker.utils.stream import Stream
from matchmaker.utils.symbolic import (
    Buffer,
    framed_midi_messages_from_performance,
    get_available_midi_port,
    midi_messages_from_performance,
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
    polling_period: Optional[float]
    midi_messages: List[Tuple[mido.Message, float]]

    def __init__(
        self,
        processor: Optional[Union[Callable, Processor]] = None,
        file_path: Optional[str] = None,
        polling_period: Optional[float] = POLLING_PERIOD,
        port: Optional[Union[MidiInputPort, str]] = None,
        queue: RECVQueue = None,
        init_time: Optional[float] = None,
        return_midi_messages: bool = False,
        mediator: Optional[CeusMediator] = None,
        virtual_port: bool = False,
    ):
        if processor is None:
            processor = PitchIOIProcessor()

        Stream.__init__(
            self,
            processor=processor,
            mock=file_path is not None,
        )
        self.file_path = file_path

        if isinstance(port, str) or port is None and file_path is None:
            port_name = get_available_midi_port(port, is_virtual=virtual_port)
            self.midi_in = mido.open_input(port_name, virtual=virtual_port)
        elif isinstance(port, MidiInputPort) and file_path is None:
            self.midi_in = port
        else:
            self.midi_in = None

        self.init_time = init_time
        self.listen = False
        self.queue = queue or RECVQueue()
        self.first_msg = False
        self.return_midi_messages = return_midi_messages
        self.mediator = mediator
        self.midi_messages = []

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

        # if output is not None:
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
                c_time = self.current_time
                self.add_midi_message(
                    msg=msg,
                    time=c_time,
                )
                self._process_frame_message(
                    data=msg,
                    c_time=c_time,
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
                    self.add_midi_message(
                        msg=msg,
                        time=c_time,
                    )
                    if msg.type in ["note_on", "note_off"]:
                        # TODO: check changing self.current_time for c_time
                        frame.append(msg, c_time)
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
            self.add_midi_message(
                msg=msg,
                time=c_time,
            )
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
        self.start_listening()
        midi_frames, frame_times = framed_midi_messages_from_performance(
            perf=self.file_path,
            polling_period=self.polling_period,
        )
        self.init_time = frame_times.min()
        for frame in midi_frames:
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
        """Start listening to midi input (open input port and get starting time)"""
        self.listen = True
        if self.mock:
            print("* Mock listening to stream....")
        else:
            print("* Start listening to MIDI stream....")
        # set initial time
        self.current_time

    def stop_listening(self):
        """Stop listening to MIDI input"""
        if self.listen:
            print("* Stop listening to MIDI stream....")
        # break while loop in self.run
        self.listen = False
        # reset init time
        self.init_time = None

        if self.midi_in is not None:
            self.midi_in.close()

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

    def add_midi_message(self, msg: mido.Message, time: float) -> None:
        self.midi_messages.append((msg, time))
