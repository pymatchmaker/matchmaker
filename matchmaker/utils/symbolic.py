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
    This method creates MIDI messages

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
