#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Miscellaneous utilities
"""

import csv
import numbers
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, Iterable, List, Optional, Union

import librosa
import numpy as np
import partitura
import scipy
import soundfile as sf
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from partitura.score import ScoreLike

from matchmaker.features.audio import SAMPLE_RATE


# Tempo marking to BPM mapping
# Reference: https://en.wikipedia.org/wiki/Tempo#Basic_tempo_markings
TEMPO_MARKING_TO_BPM = {
    # Very slow (24-40 BPM)
    "larghissimo": 24,
    "grave": 30,
    # Slow (40-66 BPM)
    "largo": 40,
    "larghetto": 50,
    "lento": 60,
    # Slow-moderate (44-80 BPM)
    "adagio": 60,
    "adagietto": 70,
    # Walking pace (56-108 BPM)
    "andante": 80,
    "andantino": 90,
    # Moderate (86-126 BPM)
    "moderato": 110,
    "allegretto": 120,
    # Fast (100-156 BPM)
    "allegro": 130,
    # Very fast (136-200+ BPM)
    "vivace": 150,
    "vivacissimo": 170,
    "presto": 180,
    "prestissimo": 200,
}


def extract_tempo_marking_from_musicxml(
    score_file: Union[str, Path],
) -> Optional[float]:
    """
    Extract tempo from text tempo marking (e.g., "Allegro", "Andante") in MusicXML.

    Parses <direction-type><words>...</words></direction-type> elements and
    matches against known tempo markings. The tempo is adjusted based on the
    time signature to return quarter-note BPM.

    Handles both simple and compound meters:
    - Simple meters (2/2, 4/4, etc.): beat is the note indicated by denominator
    - Compound meters (6/8, 9/8, 12/8): beat is a dotted note (3 subdivisions)

    For example:
    - 2/2 "Allegro" (130 half notes/min) → 260 quarter-note BPM
    - 6/8 "Presto" (180 dotted-quarters/min) → 270 quarter-note BPM

    Parameters
    ----------
    score_file : str or Path
        Path to the MusicXML file

    Returns
    -------
    float or None
        Quarter-note BPM based on tempo marking, or None if not found
    """
    try:
        tree = ET.parse(str(score_file))
        root = tree.getroot()

        # Track current time signature
        # Default to 4/4 (simple meter, quarter note beat)
        current_beats = 4
        current_beat_type = 4

        # Process measures in order to track time signature changes
        for part in root.iter("part"):
            for measure in part.iter("measure"):
                # Check for time signature change in this measure
                for attributes in measure.iter("attributes"):
                    for time_elem in attributes.iter("time"):
                        beats_elem = time_elem.find("beats")
                        beat_type_elem = time_elem.find("beat-type")
                        if beats_elem is not None and beats_elem.text:
                            current_beats = int(beats_elem.text)
                        if beat_type_elem is not None and beat_type_elem.text:
                            current_beat_type = int(beat_type_elem.text)

                # Look for tempo marking in this measure
                for direction in measure.iter("direction"):
                    for direction_type in direction.iter("direction-type"):
                        for words in direction_type.iter("words"):
                            if words.text:
                                text = words.text.lower().strip()
                                # Check each tempo marking
                                for marking, bpm in TEMPO_MARKING_TO_BPM.items():
                                    # Match if marking appears at the start of the text
                                    if re.match(rf"^{marking}\b", text):
                                        # Check if compound meter (6/8, 9/8, 12/8, etc.)
                                        # Compound meter: numerator divisible by 3 and >= 6
                                        is_compound = (
                                            current_beats >= 6
                                            and current_beats % 3 == 0
                                        )

                                        if is_compound:
                                            # Compound meter: beat is dotted note
                                            # e.g., 6/8: dotted quarter = 3 eighth notes
                                            # To maintain the same "feel" as simple meter:
                                            # - In 4/4 "Andante=80": quarter = 0.75s
                                            # - In 6/8 we want dotted quarter as beat
                                            # - dotted quarter = 1.5 × quarter
                                            # - So dotted quarter BPM = quarter BPM / 1.5
                                            quarter_note_bpm = bpm / 1.5
                                        else:
                                            # Simple meter: beat-type indicates beat note
                                            # Convert to quarter-note BPM
                                            # beat-type 2 (half note): multiply by 2
                                            # beat-type 4 (quarter note): no change
                                            # beat-type 8 (eighth note): divide by 2
                                            quarter_note_bpm = bpm * (
                                                4.0 / current_beat_type
                                            )

                                        return float(quarter_note_bpm)

                # Only process first part to avoid duplicates
                break
    except Exception:
        pass

    return None


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


def get_tempo_from_score(
    score_part: ScoreLike,
    score_file: Optional[Union[str, Path]] = None,
) -> Optional[float]:
    """
    Extract first tempo marking from score if available.

    Tries multiple sources in order:
    1. Partitura Tempo objects (explicit BPM)
    2. MusicXML <sound tempo="..."/> element (if score_file provided)
    3. Text tempo marking (e.g., "Allegro", "Andante") converted to approximate BPM

    Parameters
    ----------
    score_part : ScoreLike
        Partitura score part
    score_file : str or Path, optional
        Path to the score file. Used as fallback to parse MusicXML directly
        when partitura doesn't extract tempo.

    Returns
    -------
    float or None
        Tempo in BPM if found in score, None otherwise.
    """
    # Try partitura Tempo objects first
    if score_part is not None:
        try:
            for tempo_obj in score_part.iter_all(partitura.score.Tempo):
                if hasattr(tempo_obj, "bpm") and tempo_obj.bpm is not None:
                    return float(tempo_obj.bpm)
        except Exception:
            pass

    # Fallback: parse MusicXML directly for <sound tempo="..."/>
    if score_file is not None:
        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(str(score_file))
            root = tree.getroot()

            for sound_elem in root.iter("sound"):
                tempo_attr = sound_elem.get("tempo")
                if tempo_attr is not None:
                    return float(tempo_attr)
        except Exception:
            pass

    # Fallback: extract from text tempo marking (e.g., "Allegro", "Andante")
    if score_file is not None:
        text_tempo = extract_tempo_marking_from_musicxml(score_file)
        if text_tempo is not None:
            return text_tempo

    return None


def get_tempo_at_beat(
    score_part: ScoreLike,
    beat: float,
    default_tempo: float = 120.0,
) -> float:
    """
    Get tempo (BPM) at a specific beat position in the score.

    Uses score tempo markings if available. Falls back to default_tempo otherwise.

    Parameters
    ----------
    score_part : ScoreLike
        Partitura score part
    beat : float
        Beat position in the score
    default_tempo : float
        Default tempo to use if no tempo markings found

    Returns
    -------
    float
        Tempo in BPM at the given beat position
    """
    if score_part is None:
        return default_tempo

    # Collect all tempo markings with their positions
    tempo_changes = []
    try:
        for tempo_obj in score_part.iter_all(partitura.score.Tempo):
            if hasattr(tempo_obj, "bpm") and tempo_obj.bpm is not None:
                # Get beat position of tempo marking
                start_time = getattr(tempo_obj, "start", None)
                if start_time is not None:
                    tempo_beat = score_part.beat_map(start_time.t)
                    tempo_changes.append((tempo_beat, float(tempo_obj.bpm)))
    except Exception:
        pass

    if not tempo_changes:
        return default_tempo

    # Sort by beat position
    tempo_changes.sort(key=lambda x: x[0])

    # Find the tempo at the given beat (last tempo marking before or at beat)
    current_tempo = default_tempo
    for tempo_beat, bpm in tempo_changes:
        if tempo_beat <= beat:
            current_tempo = bpm
        else:
            break

    return current_tempo


def adjust_tempo_for_performance_audio(
    score: ScoreLike, performance_audio: Path, default_tempo: int = 120
):
    """
    Adjust the tempo of the score part to match the performance audio.
    We round up the tempo to the nearest 20 bpm to avoid too much optimization.

    Parameters
    ----------
    score : partitura.score.ScoreLike
        The score to adjust the tempo of.
    performance_audio : Path
        The performance audio file to adjust the tempo to.
    default_tempo : int
        The default tempo of the score.
    """
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


def synthesize_single_note_audio_fluidsynth(
    pitch: int,
    *,
    bpm: float,
    samplerate: int,
    duration_sec: float = 0.35,
    velocity: int = 80,
    program: int = 0,
    channel: int = 0,
    ticks_per_beat: int = 480,
) -> np.ndarray:
    """
    Create a single-note MIDI in memory and render it to audio using Partitura+FluidSynth.

    This is useful for regenerating GaussianToneModel templates from a "realistic" piano soundfont,
    instead of analytic sine/noise synthesis.
    """

    pitch = int(pitch)
    velocity = int(np.clip(int(velocity), 1, 127))
    program = int(np.clip(int(program), 0, 127))
    channel = int(np.clip(int(channel), 0, 15))
    ticks_per_beat = int(max(1, ticks_per_beat))

    # Convert duration (sec) -> ticks using requested bpm.
    dur_beats = float(duration_sec) * float(bpm) / 60.0
    dur_ticks = int(max(1, round(dur_beats * ticks_per_beat)))

    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Tempo + instrument
    track.append(
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(float(bpm)), time=0)
    )
    track.append(
        mido.Message("program_change", program=program, channel=channel, time=0)
    )

    # Note
    track.append(
        mido.Message("note_on", note=pitch, velocity=velocity, channel=channel, time=0)
    )
    track.append(
        mido.Message(
            "note_off", note=pitch, velocity=0, channel=channel, time=dur_ticks
        )
    )

    perf = partitura.load_performance_midi(mid)
    y = partitura.save_wav_fluidsynth(perf, samplerate=int(samplerate), bpm=float(bpm))
    return np.asarray(y, dtype=np.float32)


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
        plt.plot(target, ref, ".", color="cyan", alpha=0.5, markersize=3)

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
    plt.close()


def plot_and_save_gt_vs_pred_points(
    perf_annots: np.ndarray,
    perf_annots_predicted: np.ndarray,
    save_dir: Path,
    name: str,
    *,
    score_y: Optional[np.ndarray] = None,
    frame_rate: float = 1.0,
    x_unit: str = "frames",
):
    """
    Save a simple scatter plot that overlays ground-truth and predicted annotations.

    This is intentionally lightweight and does not depend on score audio or a distance matrix.

    Parameters
    ----------
    perf_annots : np.ndarray
        Ground-truth performance annotation positions (seconds).
    perf_annots_predicted : np.ndarray
        Predicted performance annotation positions (seconds).
    save_dir : Path
        Directory to save the plot.
    name : str
        Base name for the saved file (without extension).
    score_y : np.ndarray | None
        Optional y-axis values (e.g., score beats, score timeline seconds, or state indices).
        If None, uses annotation index.
    frame_rate : int | float
        Audio frame rate used to convert seconds -> frames (only used when x_unit="frames").
    x_unit : {"frames", "seconds"}
        X-axis unit for performance positions.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    gt = np.asarray(perf_annots, dtype=float)
    pred = np.asarray(perf_annots_predicted, dtype=float)
    n = min(len(gt), len(pred))
    gt = gt[:n]
    pred = pred[:n]

    if x_unit not in {"frames", "seconds"}:
        raise ValueError(f"Invalid x_unit={x_unit!r}. Use 'frames' or 'seconds'.")

    # X: performance axis
    if x_unit == "frames":
        x_gt = gt * float(frame_rate)
        x_pred = pred * float(frame_rate)
        xlabel = "performance frame index"
    else:
        x_gt = gt
        x_pred = pred
        xlabel = "performance time (s)"

    # Y: score axis
    if score_y is None:
        y = np.arange(n)
        ylabel = "score annotation index"
    else:
        y = np.asarray(score_y, dtype=float)[:n]
        ylabel = "score position (in beats)"

    fig, ax = plt.subplots(figsize=(30, 30))
    # GT: red 'x', Pred: blue dots
    ax.scatter(
        x_gt,
        y,
        label="ground truth",
        s=28,
        alpha=0.9,
        marker="x",
        color="red",
    )
    ax.scatter(
        x_pred,
        y,
        label="predicted",
        s=18,
        alpha=0.9,
        marker="o",
        color="blue",
        linewidths=0,
    )
    ax.set_title(f"[{save_dir.name}] GT vs Pred (scatter) ({name})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Keep a roughly 1:1 *figure* ratio (square canvas) without forcing equal data units,
    # since x(frames) and y(beats/states) have different scales.
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_dir / f"annots_{name}.png", dpi=150)
    plt.close(fig)


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
    # Always produce a lightweight plot that overlays GT vs Pred, even when score_audio is unavailable.
    save_dir = Path(save_dir) if save_dir is not None else Path("./tests/results")
    save_dir.mkdir(parents=True, exist_ok=True)
    run_name_suffix = (
        f"{Path(perf_file).stem}_{run_name}" if run_name else f"{Path(perf_file).stem}"
    )

    # Use score_annots as y-axis when it is the same length and monotonic (beats/seconds).
    score_y = None
    try:
        sx = np.asarray(score_annots, dtype=float)
        if sx.ndim == 1 and len(sx) == len(perf_annots) and np.all(np.diff(sx) >= 0):
            score_y = sx
    except Exception:
        score_y = None

    plot_and_save_gt_vs_pred_points(
        perf_annots=perf_annots,
        perf_annots_predicted=perf_annots_predicted,
        save_dir=save_dir,
        name=run_name_suffix,
        score_y=score_y,
        frame_rate=frame_rate,
        x_unit="frames",
    )

    # Optional: keep previous debug artifacts when inputs are available.
    if score_audio is not None:
        score_audio_dir = Path("./score_audio")
        score_audio_dir.mkdir(parents=True, exist_ok=True)
        try:
            save_mixed_audio(
                score_audio,
                score_annots,
                save_path=score_audio_dir
                / f"score_audio_{Path(score_file).parent.parent.name}_{Path(score_file).stem}_{run_name_suffix}.wav",
            )
        except Exception:
            pass

        score_predicted_audio_dir = Path("./score_audio_predicted")
        score_predicted_audio_dir.mkdir(parents=True, exist_ok=True)
        try:
            save_mixed_audio(
                score_audio,
                score_annots_predicted,
                save_path=score_predicted_audio_dir
                / f"score_audio_{Path(score_file).parent.parent.name}_{Path(score_file).parent.name}_{run_name_suffix}.wav",
            )
        except Exception:
            pass

    if perf_file is not None:
        perf_audio_dir = Path("./performance_audio")
        perf_audio_dir.mkdir(parents=True, exist_ok=True)
        try:
            save_mixed_audio(
                perf_file,
                perf_annots,
                save_path=perf_audio_dir
                / f"perf_audio_{Path(perf_file).parent.parent.name}_{Path(perf_file).parent.name}_{run_name_suffix}.wav",
            )
        except Exception:
            pass

        perf_predicted_audio_dir = Path("./performance_audio_predicted")
        perf_predicted_audio_dir.mkdir(parents=True, exist_ok=True)
        try:
            save_mixed_audio(
                perf_file,
                perf_annots_predicted,
                save_path=perf_predicted_audio_dir
                / f"perf_audio_{Path(perf_file).parent.parent.name}_{Path(perf_file).parent.name}_{run_name_suffix}.wav",
            )
        except Exception:
            pass

    # Keep the distance-matrix plot only when the needed model fields exist.
    if (
        model is not None
        and hasattr(model, "warping_path")
        and hasattr(model, "reference_features")
        and hasattr(model, "input_features")
        and hasattr(model, "distance_func")
    ):
        try:
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
        except Exception:
            pass
