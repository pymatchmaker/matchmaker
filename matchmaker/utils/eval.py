import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy.typing import NDArray

from matchmaker.features.audio import FRAME_RATE

TOLERANCES = [50, 100, 300, 500, 1000, 2000]


def transfer_positions(wp, ref_anns, frame_rate):
    """
    Transfer the positions of the reference annotations to the target annotations using the warping path.
    Parameters
    ----------
    wp : np.array with shape (2, T)
        array of warping path.
        warping_path[0] is the index of the reference (score) feature and warping_path[1] is the index of the target(input) feature.
    ref_ann : List[float]
        reference annotations in seconds.
    """
    x, y = wp[0], wp[1]
    ref_anns_frame = np.round(ref_anns * frame_rate)
    predicted_targets = np.array([y[np.where(x >= r)[0][0]] for r in ref_anns_frame])
    return predicted_targets / frame_rate


def get_evaluation_results(
    score_annots,
    perf_annots,
    warping_path,
    frame_rate,
    tolerance=TOLERANCES,
):
    target_annots_predicted = transfer_positions(
        warping_path, score_annots, frame_rate=frame_rate
    )
    errors_in_delay = (
        (perf_annots - target_annots_predicted) / frame_rate * 1000
    )  # in milliseconds

    absolute_errors_in_delay = np.abs(errors_in_delay)
    filtered_abs_errors_in_delay = absolute_errors_in_delay[
        absolute_errors_in_delay <= tolerance[-1]
    ]

    results = {
        "mean": float(f"{np.mean(filtered_abs_errors_in_delay):.4f}"),
        "median": float(f"{np.median(filtered_abs_errors_in_delay):.4f}"),
        "std": float(f"{np.std(filtered_abs_errors_in_delay):.4f}"),
        "skewness": float(f"{scipy.stats.skew(filtered_abs_errors_in_delay):.4f}"),
        "kurtosis": float(f"{scipy.stats.kurtosis(filtered_abs_errors_in_delay):.4f}"),
    }
    for tau in tolerance:
        results[f"{tau}ms"] = float(f"{np.mean(absolute_errors_in_delay <= tau):.4f}")
    results["count"] = len(filtered_abs_errors_in_delay)
    return results


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
    with open(perf_ann_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        perf_annots = [float(row[0]) for row in reader]
    for i, (ref, target) in enumerate(zip(score_annots, perf_annots)):
        plt.plot(
            target * frame_rate, ref * frame_rate, "x", color="r", alpha=1, markersize=3
        )
    plt.savefig(save_dir / f"{run_name}.png")
