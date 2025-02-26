#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Miscellaneous utilities
"""

import numbers
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Iterable, List, Union

import librosa
import numpy as np
import partitura
from partitura.io.exportmidi import get_ppq
from partitura.score import ScoreLike


class MatchmakerInvalidParameterTypeError(Exception):
    """
    Error for flagging an invalid parameter type.
    """

    def __init__(
        self,
        parameter_name: str,
        required_parameter_type: Union[type, Iterable[type]],
        actual_parameter_type: type,
        *args,
    ) -> None:
        if isinstance(required_parameter_type, Iterable):
            rqpt = ", ".join([f"{pt}" for pt in required_parameter_type])
        else:
            rqpt = required_parameter_type
        message = f"`{parameter_name}` was expected to be {rqpt}, but is {actual_parameter_type}"

        super().__init__(message, *args)


class MatchmakerInvalidOptionError(Exception):
    """
    Error for invalid option.
    """

    def __init__(self, parameter_name, valid_options, value, *args) -> None:
        rqop = ", ".join([f"{op}" for op in valid_options])
        message = f"`{parameter_name}` was expected to be in {rqop}, but is {value}"

        super().__init__(message, *args)


class MatchmakerMissingParameterError(Exception):
    """
    Error for flagging a missing parameter
    """

    def __init__(self, parameter_name: Union[str, List[str]], *args) -> None:
        if isinstance(parameter_name, Iterable) and not isinstance(parameter_name, str):
            message = ", ".join([f"`{pn}`" for pn in parameter_name])
            message = f"{message} were not given"
        else:
            message = f"`{parameter_name}` was not given."
        super().__init__(message, *args)


def ensure_rng(
    seed: Union[numbers.Integral, np.random.RandomState],
) -> np.random.RandomState:
    """
    Ensure random number generator is a np.random.RandomState instance

    Parameters
    ----------
    seed : int or np.random.RandomState
        An integer to serve as the seed for the random number generator or a
        `np.random.RandomState` instance.

    Returns
    -------
    rng : np.random.RandomState
        A random number generator.
    """

    if isinstance(seed, numbers.Integral):
        rng = np.random.RandomState(seed)
        return rng
    elif isinstance(seed, np.random.RandomState):
        rng = seed
        return rng
    else:
        raise ValueError(
            "`seed` should be an integer or an instance of "
            f"`np.random.RandomState` but is {type(seed)}"
        )


class RECVQueue(Queue):
    """
    Queue with a recv method (like Pipe)

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
            except Empty:  # pragma: no cover
                pass

    def poll(self) -> bool:
        return self.empty()


def get_window_indices(indices: np.ndarray, context: int) -> np.ndarray:
    # Create a range array from -context to context (inclusive)
    range_array = np.arange(-context, context + 1)

    # Reshape indices to be a column vector (len(indices), 1)
    indices = indices[:, np.newaxis]

    # Use broadcasting to add the range array to each index
    out_array = indices + range_array

    return out_array.astype(int)


def is_audio_file(file_path) -> bool:
    audio_extensions = {".wav", ".mp3", ".flac", ".aac", ".ogg", ".m4a"}
    ext = Path(file_path).suffix
    return ext.lower() in audio_extensions


def is_midi_file(file_path) -> bool:
    midi_extensions = {".mid", ".midi"}
    ext = Path(file_path).suffix
    return ext.lower() in midi_extensions


def interleave_with_constant(
    array: np.array,
    constant_row: float = 0,
) -> np.ndarray:
    """
    Interleave a matrix with rows of a constant value.

    Parameters
    -----------
    array : np.ndarray
    """
    # Determine the shape of the input array
    num_rows, num_cols = array.shape

    # Create an output array with interleaved rows (double the number of rows)
    interleaved_array = np.zeros((num_rows * 2, num_cols), dtype=array.dtype)

    # Set the odd rows to the original array and even rows to the constant_row
    interleaved_array[0::2] = array
    interleaved_array[1::2] = constant_row

    return interleaved_array


def adjust_tempo_for_performance_audio(score: ScoreLike, performance_audio: Path):
    """
    Adjust the tempo of the score part to match the performance audio.
    We round up the tempo to the nearest 20 bpm to avoid too much optimization.

    Parameters
    ----------
    score : partitura.score.ScoreLike
        The score to adjust the tempo of.
    performance_audio : Path
        The performance audio file to adjust the tempo to.
    """
    default_tempo = 120
    score_midi = partitura.save_score_midi(score, out=None)
    source_length = score_midi.length
    target_length = librosa.get_duration(path=str(performance_audio))
    ratio = target_length / source_length
    rounded_tempo = int(
        (default_tempo / ratio + 19) // 20 * 20
    )  # round up to nearest 20
    print(
        f"default tempo: {default_tempo} (score length: {source_length}) -> adjusted_tempo: {rounded_tempo} (perf length: {target_length})"
    )
    return rounded_tempo


def get_current_note_bpm(score: ScoreLike, onset_beat: float, tempo: float) -> float:
    """Get the adjusted BPM for a given note onset beat position based on time signature."""
    current_time = score.inv_beat_map(onset_beat)
    beat_type_changes = [
        {"start": time_sig.start, "beat_type": time_sig.beat_type}
        for time_sig in score.time_sigs
    ]

    # Find the latest applicable time signature change
    latest_change = next(
        (
            change
            for change in reversed(beat_type_changes)
            if current_time >= change["start"].t
        ),
        None,
    )

    # Return adjusted BPM if time signature change exists, else default tempo
    return latest_change["beat_type"] / 4 * tempo if latest_change else tempo


def generate_score_audio(score: ScoreLike, bpm: float, samplerate: int):
    bpm_array = [
        [onset_beat, get_current_note_bpm(score, onset_beat, bpm)]
        for onset_beat in score.note_array()["onset_beat"]
    ]
    bpm_array = np.array(bpm_array)
    score_audio = partitura.save_wav_fluidsynth(
        score,
        bpm=bpm_array,
        samplerate=samplerate,
    )

    first_onset_in_beat = score.note_array()["onset_beat"].min()
    first_onset_in_time = (
        score.inv_beat_map(first_onset_in_beat) / get_ppq(score) * (60 / bpm)
    )
    # add padding to the beginning of the score audio
    padding_size = int(first_onset_in_time * samplerate)
    score_audio = np.pad(score_audio, (padding_size, 0))

    last_onset_in_div = np.floor(score.note_array()["onset_div"].max())
    last_onset_in_time = last_onset_in_div / get_ppq(score) * (60 / bpm)

    buffer_size = 0.1  # for assuring the last onset is included (in seconds)
    last_onset_in_time += buffer_size
    score_audio = score_audio[: int(last_onset_in_time * samplerate)]
    return score_audio
