#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Utilities for symbolic music processing (e.g., MIDI)
"""
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import mido
import numpy as np
import partitura as pt
from fluidsynth import Synth
from numpy.typing import NDArray
from partitura.io.exportaudio import SAMPLE_RATE
from partitura.performance import Performance, PerformanceLike, PerformedPart
from partitura.score import ScoreLike
from partitura.utils.misc import PathLike
from partitura.utils.music import (
    ensure_notearray,
    get_time_units_from_note_array,
    performance_notearray_from_score_notearray,
)
from scipy.io import wavfile

DEFAULT_SOUNDFONT = Path("~/soundfonts/sf2/MuseScore_General.sf2").expanduser()


def synthesize_fluidsynth(
    note_info: Union[ScoreLike, PerformanceLike, np.ndarray],
    samplerate: int = SAMPLE_RATE,
    soundfont: str = DEFAULT_SOUNDFONT,
    bpm: Union[float, np.ndarray, Callable] = 60,
) -> np.ndarray:
    """
    This method will be deleted when it becomes a part of Partitura (for release 1.5.0)!
    """

    if isinstance(note_info, pt.performance.Performance):
        for ppart in note_info:
            ppart.sustain_pedal_threshold = 127

    if isinstance(note_info, pt.performance.PerformedPart):
        note_info.sustain_pedal_threshold = 127
    note_array = ensure_notearray(note_info)

    onset_unit, _ = get_time_units_from_note_array(note_array)
    if np.min(note_array[onset_unit]) <= 0:
        note_array[onset_unit] = note_array[onset_unit] + np.min(note_array[onset_unit])

    pitch = note_array["pitch"]
    # If the input is a score, convert score time to seconds
    if onset_unit != "onset_sec":
        pnote_array = performance_notearray_from_score_notearray(
            snote_array=note_array,
            bpm=bpm,
        )
        onsets = pnote_array["onset_sec"]
        offsets = pnote_array["onset_sec"] + pnote_array["duration_sec"]
        # duration = pnote_array["duration_sec"]
        channel = pnote_array["channel"]
        track = pnote_array["track"]
        velocity = pnote_array["velocity"]
    else:
        onsets = note_array["onset_sec"]
        offsets = note_array["onset_sec"] + note_array["duration_sec"]
        # duration = note_array["duration_sec"]

        if "velocity" in note_array.dtype.names:
            velocity = note_array["velocity"]
        else:
            velocity = np.ones(len(onsets), dtype=int) * 64
        if "channel" in note_array.dtype.names:
            channel = note_array["channel"]
        else:
            channel = np.zeros(len(onsets), dtype=int)

        if "track" in note_array.dtype.names:
            track = note_array["track"]
        else:
            track = np.zeros(len(onsets), dtype=int)

    controls = []
    if isinstance(note_info, pt.performance.Performance):

        for ppart in note_info:
            controls += ppart.controls

    unique_tracks = list(
        set(list(np.unique(track)) + list(set([c["track"] for c in controls])))
    )

    track_dict = defaultdict(lambda: defaultdict(list))

    for tn in unique_tracks:
        track_idxs = np.where(track == tn)[0]

        track_channels = channel[track_idxs]
        track_pitch = pitch[track_idxs]
        track_onsets = onsets[track_idxs]
        track_offsets = offsets[track_idxs]
        track_velocity = velocity[track_idxs]

        unique_channels = np.unique(track_channels)

        track_controls = [c for c in controls if c["track"] == tn]

        for chn in unique_channels:

            channel_idxs = np.where(track_channels == chn)[0]

            channel_pitch = track_pitch[channel_idxs]
            channel_onset = track_onsets[channel_idxs]
            channel_offset = track_offsets[channel_idxs]
            channel_velocity = track_velocity[channel_idxs]

            channel_controls = [c for c in track_controls if c["channel"] == chn]

            track_dict[tn][chn] = [
                channel_pitch,
                channel_onset,
                channel_offset,
                channel_velocity,
                channel_controls,
            ]

    # set to mono
    synthesizer = Synth(samplerate=SAMPLE_RATE)
    sf_id = synthesizer.sfload(soundfont)

    audio_signals = []
    for tn, channel_info in track_dict.items():

        for chn, (pi, on, off, vel, ctrls) in channel_info.items():

            audio_signal = synth_note_info(
                pitch=pi,
                onsets=on,
                offsets=off,
                velocities=vel,
                controls=ctrls,
                program=None,
                synthesizer=synthesizer,
                sf_id=sf_id,
                channel=chn,
                samplerate=samplerate,
            )
            audio_signals.append(audio_signal)

    # pad audio signals:

    signal_lengths = [len(signal) for signal in audio_signals]
    max_len = np.max(signal_lengths)

    output_audio_signal = np.zeros(max_len)

    for sl, audio_signal in zip(signal_lengths, audio_signals):

        output_audio_signal[:sl] += audio_signal

    # normalization term
    norm_term = max(audio_signal.max(), abs(audio_signal.min()))
    output_audio_signal /= norm_term

    return output_audio_signal


def synth_note_info(
    pitch: np.ndarray,
    onsets: np.ndarray,
    offsets: np.ndarray,
    velocities: np.ndarray,
    controls: Optional[list],
    program: Optional[int],
    synthesizer: Synth,
    sf_id: int,
    channel: int,
    samplerate: int = SAMPLE_RATE,
) -> np.ndarray:

    # set program
    synthesizer.program_select(channel, sf_id, 0, program or 0)

    # TODO: extend piece duration to account for pedal info.
    if len(controls) > 0 and len(offsets) > 0:
        piece_duration = max(offsets.max(), np.max([c["time"] for c in controls]))
    elif len(controls) > 0 and len(offsets) == 0:
        piece_duration = np.max([c["time"] for c in controls])
    elif len(controls) == 0 and len(offsets) > 0:
        piece_duration = offsets.max()
    else:
        # return a single zero
        audio_signal = np.zeros(1)
        return audio_signal

    num_frames = int(np.round(piece_duration * samplerate))

    # Initialize array containing audio
    audio_signal = np.zeros(num_frames, dtype="float")

    # Initialize the time axis
    x = np.linspace(0, piece_duration, num=num_frames)

    # onsets in frames (i.e., indices of the `audio_signal` array)
    onsets_in_frames = np.searchsorted(x, onsets, side="left")
    offsets_in_frames = np.searchsorted(x, offsets, side="left")

    messages = []
    for ctrl in controls or []:

        messages.append(
            (
                "cc",
                channel,
                ctrl["number"],
                ctrl["value"],
                np.searchsorted(x, ctrl["time"], side="left"),
            )
        )

    for pi, vel, oif, ofif in zip(
        pitch, velocities, onsets_in_frames, offsets_in_frames
    ):

        messages += [
            ("noteon", channel, pi, vel, oif),
            ("noteoff", channel, pi, ofif),
        ]

    # sort messages
    messages.sort(key=lambda x: x[-1])

    delta_times = [
        int(nm[-1] - cm[-1]) for nm, cm in zip(messages[1:], messages[:-1])
    ] + [0]

    for dt, msg in zip(delta_times, messages):

        msg_type = msg[0]
        msg_time = msg[-1]
        getattr(synthesizer, msg_type)(*msg[1:-1])

        samples = synthesizer.get_samples(dt)[::2]
        audio_signal[msg_time : msg_time + dt] = samples

    return audio_signal


def save_wav_fluidsynth(
    input_data: Union[ScoreLike, PerformanceLike, np.ndarray],
    out: Optional[PathLike] = None,
    samplerate: int = SAMPLE_RATE,
    soundfont: PathLike = DEFAULT_SOUNDFONT,
    bpm: Union[float, np.ndarray, Callable] = 60,
) -> Optional[np.ndarray]:
    """
    Export a score (a `Score`, `Part`, `PartGroup` or list of `Part` instances),
    a performance (`Performance`, `PerformedPart` or list of `PerformedPart` instances)
    as a WAV file using fluidsynth

    Parameters
    ----------
    input_data : ScoreLike, PerformanceLike or np.ndarray
        A partitura object with note information.

    out : PathLike or None
        Path of the output Wave file. If None, the method outputs
        the audio signal as an array (see `audio_signal` below).

    samplerate: int
        The sample rate of the audio file in Hz. The default is 44100Hz.

    soundfont : PathLike
        Path to the soundfont in SF2 format for fluidsynth.

    bpm : float, np.ndarray, callable
        The bpm to render the output (if the input is a score-like object).
        See `partitura.utils.music.performance_notearray_from_score_notearray`
        for more information on this parameter.

    Returns
    -------
    audio_signal : np.ndarray
       Audio signal as a 1D array. Only returned if `out` is None.
    """
    audio_signal = synthesize_fluidsynth(
        note_info=input_data,
        samplerate=samplerate,
        soundfont=soundfont,
        bpm=bpm,
    )

    if out is not None:
        # Write audio signal

        # convert to 16bit integers (save as PCM 16 bit)
        amplitude = np.iinfo(np.int16).max
        if abs(audio_signal).max() <= 1:
            # convert to 16bit integers (save as PCM 16 bit)
            amplitude = np.iinfo(np.int16).max
            audio_signal *= amplitude
        wavfile.write(out, samplerate, audio_signal.astype(np.int16))
    else:
        return audio_signal


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
