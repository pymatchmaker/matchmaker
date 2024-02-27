#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Utilities for symbolic music processing (e.g., MIDI)
"""
from typing import Tuple

import mido
import numpy as np
import partitura as pt
from numpy.typing import NDArray


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

    messages = []
    message_times = []
    for ppart in perf:

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
    # features: List[Callable],
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

        frames.append(
            list(
                zip(
                    midi_msgs[idxs],
                    msg_times[idxs],
                )
            )
        )

    frames_array = np.array(
        frames,
        dtype=object,
    )

    return frames_array, frame_times


def framed_midi_messages_from_midi(
    filename: str, polling_period: float
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
