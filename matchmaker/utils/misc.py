#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Miscellaneous utilities
"""

import csv
import numbers
import os
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, Iterable, List, Union

import librosa
import mido
import numpy as np
import partitura
import scipy
import soundfile as sf
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from partitura.io.exportmidi import get_ppq
from partitura.score import ScoreLike
from partitura.utils.music import performance_notearray_from_score_notearray

from matchmaker.features.audio import SAMPLE_RATE


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


def set_latency_stats(
    latency: float, latency_stats: Dict[str, float], count: int
) -> Dict[str, float]:
    latency_stats["total_latency"] += latency
    latency_stats["total_frames"] = count
    latency_stats["max_latency"] = max(latency_stats["max_latency"], latency)
    latency_stats["min_latency"] = min(latency_stats["min_latency"], latency)

    return latency_stats


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
    # score_midi = partitura.save_score_midi(score, out=None)
    # tmp_score_path = "score.mid"
    # partitura.save_score_midi(score, out=tmp_score_path)
    # source_length = mido.MidiFile(tmp_score_path).length

    sna = score.note_array()
    pna = performance_notearray_from_score_notearray(
        snote_array=sna,
        bpm=default_tempo,
    )

    source_length = np.max(pna["onset_sec"] + pna["duration_sec"])
    target_length = librosa.get_duration(path=str(performance_audio))
    ratio = target_length / source_length
    rounded_tempo = int(
        (default_tempo / ratio + 19) // 20 * 20
    )  # round up to nearest 20
    # rounded_tempo = round(default_tempo / ratio / 20) * 20
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
        score.inv_beat_map(first_onset_in_beat)
        / score.quarter_duration_map(score.inv_beat_map(first_onset_in_beat))
        * (60 / bpm)
    )
    # add padding to the beginning of the score audio
    padding_size = int(first_onset_in_time * samplerate)
    score_audio = np.pad(score_audio, (padding_size, 0))

    last_onset_in_div = np.floor(score.note_array()["onset_div"].max())
    last_onset_in_time = (
        last_onset_in_div
        / score.quarter_duration_map(score.inv_beat_map(last_onset_in_div))
        * (60 / bpm)
    )

    buffer_size = 0.1  # for assuring the last onset is included (in seconds)
    last_onset_in_time += buffer_size
    score_audio = score_audio[: int(last_onset_in_time * samplerate)]
    return score_audio


def save_nparray_to_csv(array: NDArray, save_path: str):
    with open(save_path, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerows(array)


def save_mixed_audio(
    audio: Union[np.ndarray, str, os.PathLike],
    annots: np.ndarray,
    save_path: Union[str, os.PathLike],
    sr: int = SAMPLE_RATE,
):
    if not isinstance(audio, np.ndarray):
        audio, _ = librosa.load(audio, sr=sr)

    annots_audio = librosa.clicks(
        times=annots,
        sr=sr,
        click_freq=1000,
        length=len(audio),
    )
    audio_mixed = audio + annots_audio
    sf.write(str(save_path), audio_mixed, sr, subtype="PCM_24")


def plot_and_save_score_following_result(
    wp,
    ref_features,
    input_features,
    distance_func,
    save_dir,
    score_annots,
    perf_annots,
    frame_rate,
    name=None,
):
    xmin = 0  # performance range
    xmax = None
    ymin = 0  # score range
    ymax = None

    xmax = xmax if xmax is not None else input_features.shape[0] - 1
    ymax = ymax if ymax is not None else ref_features.shape[0] - 1
    x_indices = range(xmin, xmax + 1)
    y_indices = range(ymin, ymax + 1)

    run_name = name or "results"
    save_path = save_dir / f"wp_{run_name}.tsv"
    save_nparray_to_csv(wp.T, save_path.as_posix())

    dist = scipy.spatial.distance.cdist(
        ref_features[y_indices, :],
        input_features[x_indices, :],
        metric=distance_func,
    )  # [d, wy]
    plt.figure(figsize=(10, 10))
    plt.imshow(
        dist,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=(xmin, xmax, ymin, ymax),
    )
    mask_perf = (xmin <= perf_annots * frame_rate) & (perf_annots * frame_rate <= xmax)
    mask_score = (ymin <= score_annots * frame_rate) & (
        score_annots * frame_rate <= ymax
    )
    plt.title(
        f"[{save_dir.name}/{run_name}] \n Matchmaker alignment path with ground-truth labels",
        fontsize=15,
    )
    plt.xlabel("Performance Features", fontsize=15)
    plt.ylabel("Score Features", fontsize=15)

    # plot online DTW path
    cropped_history = [
        (ref, target)
        for (ref, target) in wp.T
        if xmin <= target <= xmax and ymin <= ref <= ymax
    ]
    for ref, target in cropped_history:
        plt.plot(target, ref, ".", color="lime", alpha=0.5, markersize=3)

    # plot ground-truth labels
    for ref, target in zip(score_annots, perf_annots):
        if (xmin <= target * frame_rate <= xmax) and (ymin <= ref * frame_rate <= ymax):
            plt.plot(
                target * frame_rate,
                ref * frame_rate,
                "x",
                color="r",
                alpha=1,
                markersize=3,
                markeredgewidth=3,
            )
    plt.savefig(save_dir / f"{run_name}.png")


def save_debug_results(
    score_file,
    score_audio,
    score_annots,
    score_annots_predicted,
    perf_file,
    perf_annots,
    perf_annots_predicted,
    model,
    frame_rate,
    save_dir=None,
    run_name=None,
):
    # save score audio with beat annotations
    score_audio_dir = Path("./score_audio")
    score_audio_dir.mkdir(parents=True, exist_ok=True)
    run_name_suffix = (
        f"{Path(perf_file).stem}_{run_name}" if run_name else f"{Path(perf_file).stem}"
    )
    save_mixed_audio(
        score_audio,
        score_annots,
        save_path=score_audio_dir
        / f"score_audio_{Path(score_file).parent.parent.name}_{Path(score_file).stem}_{run_name_suffix}.wav",
    )
    # save performance audio with beat annotations
    perf_audio_dir = Path("./performance_audio")
    perf_audio_dir.mkdir(parents=True, exist_ok=True)
    save_mixed_audio(
        perf_file,
        perf_annots,
        save_path=perf_audio_dir
        / f"perf_audio_{Path(perf_file).parent.parent.name}_{Path(perf_file).parent.name}_{run_name_suffix}.wav",
    )
    # save score audio with predicted beat annotations
    score_predicted_audio_dir = Path("./score_audio_predicted")
    score_predicted_audio_dir.mkdir(parents=True, exist_ok=True)
    save_mixed_audio(
        score_audio,
        score_annots_predicted,
        save_path=score_predicted_audio_dir
        / f"score_audio_{Path(score_file).parent.parent.name}_{Path(score_file).parent.name}_{run_name_suffix}.wav",
    )
    # save performance audio with predicted beat annotations
    perf_predicted_audio_dir = Path("./performance_audio_predicted")
    perf_predicted_audio_dir.mkdir(parents=True, exist_ok=True)
    save_mixed_audio(
        perf_file,
        perf_annots_predicted,
        save_path=perf_predicted_audio_dir
        / f"perf_audio_{Path(perf_file).parent.parent.name}_{Path(perf_file).parent.name}_{run_name_suffix}.wav",
    )
    # save score following plot result
    save_dir = save_dir or Path("./tests/results")
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_and_save_score_following_result(
        model.warping_path,
        model.reference_features,
        model.input_features,
        model.distance_func,
        save_dir,
        score_annots,
        perf_annots,
        frame_rate,
        name=run_name,
    )
