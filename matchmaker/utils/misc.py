#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Miscellaneous utilities
"""

import csv
import numbers
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Union

import librosa
import mido
import numpy as np
import partitura
import scipy
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from partitura.score import ScoreLike

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


def _beats_to_frames(
    beats: np.ndarray,
    ref_frame_to_beat: np.ndarray,
) -> np.ndarray:
    """Convert beat positions to (float) frame indices via inverse interpolation."""
    frames = np.arange(len(ref_frame_to_beat), dtype=float)
    return np.interp(beats, ref_frame_to_beat, frames)


def plot_alignment(
    alignment_path: np.ndarray,
    perf_annots: np.ndarray,
    perf_annots_predicted: np.ndarray,
    save_dir: Path,
    name: str,
    score_y: Optional[np.ndarray] = None,
    frame_rate: float = 1.0,
    score_positions: Optional[np.ndarray] = None,
    ref_features: Optional[np.ndarray] = None,
    input_features: Optional[np.ndarray] = None,
    distance_func=None,
    ref_frame_to_beat: Optional[np.ndarray] = None,
):
    """Plot alignment path, GT annotations, and predicted points."""
    save_dir.mkdir(parents=True, exist_ok=True)
    gt = np.asarray(perf_annots, dtype=float)
    pred = np.asarray(perf_annots_predicted, dtype=float)
    n = min(len(gt), len(pred))
    gt, pred = gt[:n], pred[:n]

    fig, ax = plt.subplots(figsize=(30, 30))

    # Distance matrix background
    show_dist = False
    if (
        ref_features is not None
        and input_features is not None
        and distance_func is not None
    ):
        try:
            if isinstance(distance_func, str):
                dist = scipy.spatial.distance.cdist(
                    ref_features, input_features, metric=distance_func
                )
            else:
                dist = np.array(
                    [
                        [distance_func(r, i) for i in input_features]
                        for r in ref_features
                    ],
                    dtype=np.float32,
                )
            n_input = input_features.shape[0]
            n_ref = ref_features.shape[0]
            ax.imshow(
                dist,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                extent=(0, n_input - 1, 0, n_ref - 1),
            )
            show_dist = True
        except Exception:
            pass

    # x-axis: performance time in frames
    x_gt = gt * float(frame_rate)
    wp_x = alignment_path[1] * float(frame_rate)

    # y-axis: score position (beats)
    wp_in_beats = np.issubdtype(alignment_path[0].dtype, np.floating)
    if score_positions is not None and not wp_in_beats:
        wp_y = score_positions[alignment_path[0]]
    elif show_dist and wp_in_beats and ref_frame_to_beat is not None:
        wp_y = _beats_to_frames(alignment_path[0], ref_frame_to_beat)
    else:
        wp_y = alignment_path[0]

    # GT score positions (y-axis for annotation dots)
    if score_y is not None:
        y_gt = np.asarray(score_y, dtype=float)[:n]
        if show_dist and wp_in_beats and ref_frame_to_beat is not None:
            y_gt = _beats_to_frames(y_gt, ref_frame_to_beat)
    else:
        y_gt = np.arange(n)

    # Predicted score positions at GT perf times (perf→score direction)
    wp_x_sorted = np.asarray(wp_x, dtype=float)
    wp_y_sorted = np.asarray(wp_y, dtype=float)
    if len(wp_x_sorted) > 1:
        y_pred = np.interp(x_gt, wp_x_sorted, wp_y_sorted)
    else:
        y_pred = y_gt

    # Plot layers
    ax.plot(
        wp_x,
        wp_y,
        ".",
        color="white" if show_dist else "lime",
        alpha=0.7 if show_dist else 0.5,
        markersize=15,
        label="alignment path",
        zorder=2,
    )
    ax.scatter(
        x_gt,
        y_pred,
        label="predicted",
        s=80,
        alpha=0.9,
        marker="o",
        color="blue",
        linewidths=0,
        zorder=3,
    )
    ax.scatter(
        x_gt,
        y_gt,
        label="ground truth",
        s=120,
        alpha=0.9,
        marker="x",
        color="red",
        linewidths=3,
        zorder=4,
    )

    if show_dist:
        ax.set_xlim(0, input_features.shape[0] - 1)
        ax.set_ylim(0, ref_features.shape[0] - 1)

    # Beat tick labels when projected to frame space
    if show_dist and wp_in_beats and ref_frame_to_beat is not None:
        finite_beats = ref_frame_to_beat[np.isfinite(ref_frame_to_beat)]
        beat_min, beat_max = (
            finite_beats[0],
            finite_beats[-1] if len(finite_beats) > 0 else (0, 1),
        )
        n_ticks = max(2, min(12, int(beat_max - beat_min) + 1))
        beat_ticks = np.unique(
            np.round(np.linspace(beat_min, beat_max, n_ticks)).astype(int)
        )
        ax.set_yticks(_beats_to_frames(beat_ticks.astype(float), ref_frame_to_beat))
        ax.set_yticklabels([str(b) for b in beat_ticks])

    ax.set_xlabel("performance frame")
    ax.set_ylabel("score position (beats)")
    ax.set_title(f"[{save_dir.name}] alignment ({name})")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_dir / f"{name}.png", dpi=150)
    plt.close(fig)


def save_debug_results(
    alignment_path: np.ndarray,
    score_annots: np.ndarray,
    perf_annots: np.ndarray,
    perf_annots_predicted: np.ndarray,
    eval_results: dict,
    frame_rate: float,
    save_dir: Path,
    run_name: str = "results",
    score_positions: Optional[np.ndarray] = None,
    ref_features: Optional[np.ndarray] = None,
    input_features: Optional[np.ndarray] = None,
    distance_func=None,
    ref_frame_to_beat: Optional[np.ndarray] = None,
    make_plot: bool = True,
):
    """Save debug outputs: alignment path TSV, results JSON, and (optional) plot."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Alignment path TSV + results JSON + GT annotations
    save_nparray_to_csv(alignment_path.T, (save_dir / f"wp_{run_name}.tsv").as_posix())
    gt_pairs = np.column_stack([score_annots, perf_annots])
    save_nparray_to_csv(gt_pairs, (save_dir / f"gt_{run_name}.tsv").as_posix())
    import json

    with open(save_dir / f"{run_name}.json", "w") as f:
        json.dump(eval_results, f, indent=4)

    if not make_plot:
        return

    # 2. Alignment plot
    # score_y = beat positions for each annotation (y-axis of the plot)
    sx = np.asarray(score_annots, dtype=float)
    score_y = (
        sx
        if sx.ndim == 1 and len(sx) == len(perf_annots) and np.all(np.diff(sx) >= 0)
        else None
    )
    plot_alignment(
        alignment_path,
        perf_annots,
        perf_annots_predicted,
        save_dir,
        run_name,
        score_y=score_y,
        frame_rate=frame_rate,
        score_positions=score_positions,
        ref_features=ref_features,
        input_features=input_features,
        distance_func=distance_func,
        ref_frame_to_beat=ref_frame_to_beat,
    )
