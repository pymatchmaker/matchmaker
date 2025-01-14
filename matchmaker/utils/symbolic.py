#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Utilities for symbolic music processing (e.g., MIDI)
"""

from typing import List, Optional, Tuple, Union

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
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        # Logic to return the next item
        if self.index < len(self.frame):
            result = self.frame[self.index]
            self.index += 1
            return result
        else:
            # Raises StopIteration when the iteration is complete
            raise StopIteration

    def append(self, input: mido.Message, time: float) -> None:
        self.frame.append((input, time))

    # def set_start(self) -> None:
    #     if len(self.frame) > 0:
    #         self.start = np.min([time for _, time in self.frame])

    def reset(self, time: float) -> None:
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


def midi_messages_from_performance(
    perf: Union[PerformanceLike, str],
) -> Tuple[NDArray, NDArray]:
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

    if isinstance(perf, str):
        # from a MIDI/Match file
        perf = pt.load_performance(perf)

    elif isinstance(perf, np.ndarray):
        # From a Note array
        perf = PerformedPart.from_note_array(perf)

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
    start_times = np.arange(n_frames) * polling_period

    frames = []

    for cursor, s_time in enumerate(start_times):
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
        buffer.start = s_time
        frames.append(buffer)

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


class MidiDeviceInfo(object):
    """Info about a MIDI device

    See `matchmaker.utils.audio.AudioDeviceInfo`

    Parameters
    ----------
    name : str
        Name of the MIDI device
    device_index : int
        Index of the MIDI device (this is to have a similar interface as audio
        devices)
    has_input: bool
        Whether the MIDI device has inputs
    has_output: bool
        Whether the MIDI device has outputs.
    """

    def __init__(
        self,
        name: str,
        device_index: int,
        has_input: bool,
        has_output: bool,
    ) -> None:
        self.name = name
        self.device_index = device_index
        self.has_input = has_input
        self.has_output = has_output

    def __str__(self) -> str:
        out_str = (
            f"MIDI Device {self.device_index}: {self.name}\n"
            f"  - Input: {self.has_input}\n"
            f"  - Output: {self.has_output}\n"
        )

        return out_str


def get_midi_devices() -> List[MidiDeviceInfo]:
    """Get list of MIDI devices
    Returns
    -------
    midi_devices : List[MidiDeviceInfo]
        List of available MIDI devices
    """

    available_in_ports = mido.get_input_names()

    available_out_ports = mido.get_output_names()

    all_devices = list(set(available_in_ports + available_out_ports))
    all_devices.sort()

    midi_devices = []
    for i, device in enumerate(all_devices):
        has_input = device in available_in_ports
        has_output = device in available_out_ports

        midi_device = MidiDeviceInfo(
            name=device,
            device_index=i,
            has_input=has_input,
            has_output=has_output,
        )

        midi_devices.append(midi_device)

    return midi_devices


def get_available_midi_port(port: str = None, is_virtual: bool = False) -> str:
    """
    Get the available MIDI port. If a port is specified, check if it is available.

    Parameters
    ----------
    port : str, optional
        Name of the MIDI port (default is None).

    Returns
    -------
    MidiInputPort
        Available MIDI input port

    Raises
    ------
    RuntimeError
        If no MIDI input ports are available.
    ValueError
        If the specified MIDI port is not available.
    """

    if port is None and is_virtual:
        raise ValueError("Cannot open unspecified virtual port!")
    input_names = mido.get_input_names()
    if not input_names and not is_virtual:
        raise RuntimeError("No MIDI input ports available")

    if port is None and not is_virtual:
        return input_names[0]
    elif port in input_names or is_virtual:
        return port
    else:
        raise ValueError(
            f"Specified MIDI port '{port}' is not available. Available ports: {input_names}"
        )


def panic_button() -> None:
    """
    Reset all output ports
    """
    output_ports = mido.get_output_names()

    for pn in output_ports:
        with mido.open_output(pn) as outport:
            print(f"Resetting port {pn}")
            outport.reset()
        outport.close()
