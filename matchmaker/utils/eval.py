import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
    frame_rate : int
        frame rate of the audio.

    Returns
    -------
    predicted_targets : np.array with shape (T,)
        predicted target positions in seconds.
    """
    # Causal nearest
    x, y = wp[0], wp[1]
    ref_anns_frame = np.round(ref_anns * frame_rate)
    predicted_targets = []

    for r in ref_anns_frame:
        past_indices = np.where(x <= r)[0]
        if past_indices.size > 0:
            nearest_past_idx = past_indices[-1]
            predicted_targets.append(y[nearest_past_idx])

    return np.array(predicted_targets) / frame_rate

    # 1. nearest interpolation
    # ref_anns_frame = np.round(ref_anns * frame_rate)
    # positions_1_transferred_to_2 = scipy.interpolate.interp1d(
    #     wp[0], wp[1], kind="nearest"
    # )(ref_anns_frame)
    # return positions_1_transferred_to_2 / frame_rate

    # 2. threshold-crossing
    # x, y = wp[0], wp[1]
    # ref_anns_frame = np.round(ref_anns * frame_rate)
    # predicted_targets = np.array(
    #     [
    #         y[np.where(x >= r)[0][0]]
    #         for r in ref_anns_frame
    #         if np.where(x >= r)[0].size > 0
    #     ]
    # )
    # return predicted_targets / frame_rate


def get_evaluation_results(
    perf_annots,
    perf_annots_predicted,
    tolerances,
):
    errors_in_delay = (perf_annots - perf_annots_predicted) * 1000  # in milliseconds

    absolute_errors_in_delay = np.abs(errors_in_delay)
    filtered_abs_errors_in_delay = absolute_errors_in_delay[
        absolute_errors_in_delay <= tolerances[-1]
    ]

    results = {
        "mean": float(f"{np.mean(filtered_abs_errors_in_delay):.4f}"),
        "median": float(f"{np.median(filtered_abs_errors_in_delay):.4f}"),
        "std": float(f"{np.std(filtered_abs_errors_in_delay):.4f}"),
        "skewness": float(f"{scipy.stats.skew(filtered_abs_errors_in_delay):.4f}"),
        "kurtosis": float(f"{scipy.stats.kurtosis(filtered_abs_errors_in_delay):.4f}"),
    }
    for tau in tolerances:
        results[f"{tau}ms"] = float(f"{np.mean(absolute_errors_in_delay <= tau):.4f}")
    results["count"] = len(filtered_abs_errors_in_delay)
    return results
