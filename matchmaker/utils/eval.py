import csv
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import mido
import numpy as np
import pandas as pd
import partitura
import scipy
from numpy.typing import NDArray
from partitura import save_wav_fluidsynth

from matchmaker.features.audio import FRAME_RATE, SAMPLE_RATE

TOLERANCES = [50, 100, 300, 500, 1000, 2000]


def transfer_positions(wp, ref_anns, frame_rate):
    """
    Transfer the positions of the reference annotations to the target annotations using the warping path.
    Parameters
    ----------
    wp : np.array with shape (2, T)
        array of warping path.
    ref_ann : List[float]
        reference annotations in frame indices.
    """
    x, y = wp[0], wp[1]
    ref_anns_frame = np.round(ref_anns * frame_rate)
    predicted_targets = np.array([y[np.where(x >= r)[0][0]] for r in ref_anns_frame])
    return predicted_targets / frame_rate


def adjust_tempo_for_performance_audio(score_part, performance_audio: Path):
    default_tempo = 120
    score_midi = partitura.save_score_midi(score_part, out=None)
    source_length = score_midi.length
    target_length = librosa.get_duration(path=str(performance_audio))
    ratio = target_length / source_length
    # ratio = np.clip(ratio, 0.5, 2.0)  # limit the ratio to 0.5 to 2.0
    # rounded_tempo = int((default_tempo / ratio + 10) // 20 * 20)  # round to nearest 20
    rounded_tempo = int(
        (default_tempo / ratio + 19) // 20 * 20
    )  # round up to nearest 20
    print(
        f"default tempo: {default_tempo} (score length: {source_length}) -> adjusted_tempo: {rounded_tempo} (perf length: {target_length})"
    )
    return rounded_tempo


def save_nparray_to_csv(array: NDArray, save_path: str):
    with open(save_path, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerows(array)


def save_score_following_result(
    model, save_dir, score_annots, perf_ann_path: Path, frame_rate=FRAME_RATE, name=None
):
    run_name = name or "results"
    save_path = save_dir / f"wp_{run_name}.tsv"
    save_nparray_to_csv(model.warping_path.T, save_path.as_posix())

    dist = scipy.spatial.distance.cdist(
        model.reference_features,
        model.input_features[: model.warping_path[1][-1]],
        metric=model.distance_func,
    )  # [d, wy]
    plt.figure(figsize=(15, 15))
    plt.imshow(dist, aspect="auto", origin="lower", interpolation="nearest")
    plt.title(
        f"[{save_dir.name}] \n Matchmaker alignment path with ground-truth labels",
        fontsize=25,
    )
    plt.xlabel("Performance Audio frame", fontsize=15)
    plt.ylabel("Score Audio frame", fontsize=15)

    # plot online DTW path
    ref_paths, target_paths = model.warping_path[0], model.warping_path[1]
    for n in range(len(ref_paths)):
        plt.plot(
            target_paths[n], ref_paths[n], ".", color="purple", alpha=0.5, markersize=3
        )

    # plot ground-truth labels
    perf_annots = pd.read_csv(
        filepath_or_buffer=perf_ann_path, delimiter="\t", header=None
    )[0]
    for i, (ref, target) in enumerate(zip(score_annots, perf_annots)):
        # if i % 5 != 0:
        #     continue
        plt.plot(
            target * frame_rate, ref * frame_rate, "x", color="r", alpha=1, markersize=3
        )
    plt.savefig(save_dir / f"{run_name}.png")
