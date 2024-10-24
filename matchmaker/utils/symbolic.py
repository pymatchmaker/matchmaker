#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Utilities for symbolic music processing (e.g., MIDI)
"""
from typing import List, Optional, Tuple

import mido
import numpy as np
import partitura as pt
from numpy.typing import NDArray
from partitura.performance import Performance, PerformanceLike, PerformedPart

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

    def __len__(self) -> int:
        return len(self.frame)

    def append(self, input, time) -> None:
        self.frame.append((input, time))

    def set_start(self) -> None:
        if len(self.frame) > 0:
            self.start = np.min([time for _, time in self.frame])

    def reset(self, time) -> None:
        self.frame = []
        self.start = time

    @property
    def end(self) -> float:
        """
        Maximal end time of the frame
        """
        return self.start + self.polling_period

    @property
    def time(self) -> float:
        """
        Time of the middle of the frame
        """
        return self.start + 0.5 * self.polling_period

    def __len__(self) -> int:
        """
        Number of MIDI messages in the frame
        """
        return len(self.frame)

    def __str__(self) -> str:
        return str(self.frame)
        
def midi_messages_from_midi(filename: str) -> Tuple[NDArray, NDArray]:
    """
    Get a list of MIDI messages and message times from
    a MIDI file.

    The method ignores Meta messages, since they
    are not "streamed" live (see documentation for
    mido.Midifile.play)

    Parameters
    ----------
    filename : str
        The filename of the MIDI file.

    Returns
    -------
    message_array : np.ndarray of mido.Message
        An array containing MIDI messages

    message_times : np.ndarray
        An array containing the times of the messages
        in seconds.
    """
    perf = pt.load_performance(filename=filename)

    message_array, message_times_array = midi_messages_from_performance(perf=perf)

    return message_array, message_times_array


def midi_messages_from_performance(perf: PerformanceLike) -> Tuple[NDArray, NDArray]:
    """
    Get a list of MIDI messages and message times from
    a PerformedPart or a Performance object.

    The method ignores Meta messages, since they
    are not "streamed" live (see documentation for
    mido.Midifile.play)

    Parameters
    ----------
    perf : PerformanceLike
        A partitura PerformedPart or Performance object.

    Returns
    -------
    message_array : np.ndarray of mido.Message
        An array containing MIDI messages

    message_times : np.ndarray
        An array containing the times of the messages
        in seconds.
    """

    if isinstance(perf, Performance):
        pparts = perf.performedparts
    elif isinstance(perf, PerformedPart):
        pparts = [perf]

    messages = []
    message_times = []
    for ppart in pparts:

        # Get note on and note off info
        for note in ppart.notes:
            channel = note.get("channel", 0)
            note_on = mido.Message(
                type="note_on",
                note=note["pitch"],
                velocity=note["velocity"],
                channel=channel,
            )
            note_off = mido.Message(
                type="note_off",
                note=note["pitch"],
                velocity=0,
                channel=channel,
            )
            messages += [
                note_on,
                note_off,
            ]
            message_times += [
                note["note_on"],
                note["note_off"],
            ]

        # get control changes
        for control in ppart.controls:
            channel = control.get("channel", 0)
            msg = mido.Message(
                type="control_change",
                control=int(control["number"]),
                value=int(control["value"]),
                channel=channel,
            )
            messages.append(msg)
            message_times.append(control["time"])

        # Get program changes
        for program in ppart.programs:
            channel = program.get("channel", 0)
            msg = mido.Message(
                type="program_change",
                program=int(program["program"]),
                channel=channel,
            )
            messages.append(msg)
            message_times.append(program["time"])

    message_array = np.array(messages)
    message_times_array = np.array(message_times)

    sort_idx = np.argsort(message_times_array)
    # sort messages by time
    message_array = message_array[sort_idx]
    message_times_array = message_times_array[sort_idx]

    return message_array, message_times_array


def midi_messages_to_framed_midi(
    midi_msgs: NDArray,
    msg_times: NDArray,
    polling_period: float,
) -> Tuple[NDArray, NDArray]:
    """
    Convert a list of MIDI messages to a framed MIDI representation
    Parameters
    ----------
    midi_msgs: list of mido.Message
        List of MIDI messages.

    msg_times: list of float
        List of times (in seconds) at which the MIDI messages were received.

    polling_period:
        Polling period (in seconds) used to convert the MIDI messages.

    Returns
    -------
    frames_array: np.ndarray
        An array of MIDI frames.
    frame_times:
    """
    n_frames = int(np.ceil(msg_times.max() / polling_period))
    frame_times = (np.arange(n_frames) + 0.5) * polling_period

    frames = []

    for cursor in range(n_frames):
        
        buffer = Buffer(polling_period)
        if cursor == 0:
            # do not leave messages starting at 0 behind!
            idxs = np.where(msg_times <= polling_period)[0]
        else:
            idxs = np.where(
                np.logical_and(
                    msg_times > cursor * polling_period,
                    msg_times <= (cursor + 1) * polling_period,
                )
            )[0]

        buffer.frame = list(
                zip(
                    midi_msgs[idxs],
                    msg_times[idxs],
                )
            )
        buffer.set_start()
        frames.append(buffer)
        # frames.append(
        #     list(
        #         zip(
        #             midi_msgs[idxs],
        #             msg_times[idxs],
        #         )
        #     )
        # )

    frames_array = np.array(
        frames,
        dtype=object,
    )

    return frames_array, frame_times


def framed_midi_messages_from_midi(
    filename: str,
    polling_period: float,
) -> Tuple[NDArray, NDArray]:
    """
    Get a list of framed MIDI messages and frame times from
    a MIDI file.

    This is a convenience method
    """

    midi_messages, message_times = midi_messages_from_midi(
        filename=filename,
    )

    frames_array, frame_times = midi_messages_to_framed_midi(
        midi_msgs=midi_messages,
        msg_times=message_times,
        polling_period=polling_period,
    )

    return frames_array, frame_times


def framed_midi_messages_from_performance(
    perf: PerformanceLike,
    polling_period: float,
) -> Tuple[NDArray, NDArray]:
    """
    Get a list of framed MIDI messages and frame times from
    a partitura Performance or PerformedPart object.

    This is a convenience method
    """
    midi_messages, message_times = midi_messages_from_performance(perf=perf)

    frames_array, frame_times = midi_messages_to_framed_midi(
        midi_msgs=midi_messages,
        msg_times=message_times,
        polling_period=polling_period,
    )

    return frames_array, frame_times
