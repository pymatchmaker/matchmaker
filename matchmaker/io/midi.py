#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Input MIDI stream
"""
import multiprocessing
import os
import tempfile
import threading
import time
from multiprocessing import Pipe
from queue import Empty, Queue

from typing import Any, Tuple, List, Union, Optional

import mido

# Default polling period (in seconds)
POLLING_PERIOD = 0.01


def dummy_pipeline(inputs: Any) -> Any:
    """
    Dummy pipeline, which always returns the inputs

    Parameters
    ----------
    inputs : Any
        The inputs to the pipeline

    Returns
    -------
    inputs : Any
        The unmodified inputs
    """
    return inputs


class RECVQueue(Queue):
    """
    RECVQueue
    
    This class uses python's Queue.get with a timeout makes it interruptable via KeyboardInterrupt
    and even for the future where that is possibly out-dated, the interrupt can happen after each timeout
    so periodically query the queue with a timeout of 1s each attempt, finding a middleground
    between busy-waiting and uninterruptable blocked waiting
    """
    def __init__(self) -> None:
        Queue.__init__(self)


    def recv(self) -> Any:
        """
        Return and remove an item from the queue.
        """
        while True:
            try:
                return self.get(timeout=1)
            except Empty:
                pass

    def poll(self) -> bool:
        return self.empty()


class MidiInputProcess(multiprocessing.Process):
    def __init__(
        self,
        port,
        pipe,
        init_time=None,
        pipeline=None,
        return_midi_messages=False,
        output_fn=None,
        mediator=None,
    ):
        multiprocessing.Process.__init__(self)

        self.init_time = init_time
        self.listen = False
        self.pipe = pipe
        self.pipeline = pipeline
        self.first_msg = False
        if pipeline is None:
            self.pipeline = dummy_pipeline
        self.return_midi_messages = return_midi_messages

        self.lock = multiprocessing.RLock()
        self.midi_in = port
        # List to store MIDI messages
        self.midi_messages = []
        self.mediator = mediator  # TODO mediator currently not usable for processes

        if output_fn is None:
            self.output_fn = os.path.join(tempfile.gettempdir(), "input_midi.mid")
            print(f"Output will be saved in {self.output_fn}")
        else:
            self.output_fn = output_fn

    def run(self):
        self.start_listening()

        while self.listen:
            print("midi input listening")
            msg = self.midi_in.poll()
            if msg is not None:
                c_time = self.current_time
                # Append MIDI Messages to the list to export the MIDI later
                self.midi_messages.append((msg, c_time))
                # To have the same output as other MidiThreads
                # We should think of a less convoluted way to do
                # this in general
                output = self.pipeline([(msg, c_time)], c_time)
                if self.return_midi_messages:
                    self.pipe.send((msg, c_time), output)
                else:
                    self.pipe.send(output)

    @property
    def current_time(self):
        """
        Get current time since starting to listen
        """
        return time.time() - self.init_time

    def start_listening(self):
        """
        Start listening to midi input
        get starting time
        """
        self.listen = True
        if self.init_time is None:
            self.init_time = time.time()

    def stop_listening(self):
        """
        Stop listening to MIDI input
        """
        # break while loop in self.run
        with self.lock:
            self.listen = False
            # reset init time

        # close port
        # TODO!!!

        # self.terminate()
        # Join thread
        # self.join()

    def save_midi(self):
        # sort MIDI messages
        self.midi_messages.sort(key=lambda x: x[1])
        # save the MIDI file here
        # mido.MidiFile(filename=self.output_fn)


class MidiInputThread(threading.Thread):
    def __init__(
        self,
        port,
        queue,
        init_time=None,
        pipeline=None,
        return_midi_messages=False,
        mediator=None,
    ):
        threading.Thread.__init__(self)

        self.midi_in = port
        self.init_time = init_time
        self.listen = False
        self.queue = queue
        self.pipeline = pipeline
        self.first_msg = False
        if pipeline is None:
            self.pipeline = dummy_pipeline
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


class FramedMidiInputProcess(MidiInputProcess):
    def __init__(
        self,
        port,
        pipe,
        polling_period=POLLING_PERIOD,
        init_time=None,
        pipeline=None,
        return_midi_messages=False,
        mediator=None,
    ):
        super().__init__(
            port=port,
            pipe=pipe,
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

        # sttime = time.time()
        self.start_listening()
        frame = Buffer(self.polling_period)
        frame.start = self.current_time
        # TODO: Adapt from midi_online to allow for variable polling
        # TODO mediator currently not usable for processes
        # periods?
        while self.listen:
            # added if to check once again after sleep
            # TODO verify if still correct
            c_time = self.current_time
            msg = self.midi_in.poll()
            if msg is not None:
                frame.append(msg, c_time)
                if not self.first_msg:
                    self.first_msg = True
            if c_time >= frame.end and self.first_msg:
                output = self.pipeline((frame.frame, c_time))
                # self.pipe.send(output)
                if self.return_midi_messages:
                    self.pipe.send((frame.frame, output))
                else:
                    self.pipe.send(output)
                frame.reset(c_time)


class FramedMidiInputThread(MidiInputThread):
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


def create_midi_poll(
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
