from typing import TypedDict

import numpy as np
import scipy

TOLERANCES_IN_MILLISECONDS = [50, 100, 300, 500, 1000, 2000]
TOLERANCES_IN_BEATS = [0.05, 0.1, 0.3, 0.5, 1, 2]


def transfer_positions(wp, ref_anns, frame_rate, reverse=False):
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
    # Causal nearest neighbor interpolation
    if reverse:
        x, y = wp[1], wp[0]
    else:
        x, y = wp[0], wp[1]
    ref_anns_frame = np.round(ref_anns * frame_rate)
    predicted_targets = np.ones(len(ref_anns)) * np.nan

    for i, r in enumerate(ref_anns_frame):
        # 1) Scan all x values less than or equal to r and find the largest x value
        past_indices = np.where(x <= r)[0]
        if past_indices.size > 0:
            # Find indices corresponding to the largest x value
            max_x_val = x[past_indices[-1]]
            max_x_indices = np.where(x == max_x_val)[0]

            # 2) Among all y values mapped to this x value, select the minimum y value
            corresponding_y_values = y[max_x_indices]
            min_y_val = np.min(corresponding_y_values)

            # predicted_targets.append(min_y_val)
            predicted_targets[i] = min_y_val

    return np.array(predicted_targets) / frame_rate


def transfer_from_score_to_predicted_perf(wp, score_annots, frame_rate):
    predicted_perf_idx = transfer_positions(wp, score_annots, frame_rate)
    return predicted_perf_idx


def transfer_from_perf_to_predicted_score(wp, perf_annots, frame_rate):
    predicted_score_idx = transfer_positions(wp, perf_annots, frame_rate, reverse=True)
    return predicted_score_idx


def get_evaluation_results(
    gt_annots,
    predicted_annots,
    total_length,
    tolerances=TOLERANCES_IN_MILLISECONDS,
    in_seconds=True,
):
    if in_seconds:
        errors_in_delay = (gt_annots - predicted_annots) * 1000  # in milliseconds
    else:
        errors_in_delay = gt_annots - predicted_annots

    filtered_errors_in_delay = errors_in_delay[
        np.abs(errors_in_delay) <= tolerances[-1]
    ]
    filtered_abs_errors_in_delay = np.abs(filtered_errors_in_delay)

    results = {
        "mean": float(f"{np.nanmean(filtered_abs_errors_in_delay):.4f}"),
        "median": float(f"{np.nanmedian(filtered_abs_errors_in_delay):.4f}"),
        "std": float(f"{np.nanstd(filtered_abs_errors_in_delay):.4f}"),
        "skewness": float(f"{scipy.stats.skew(filtered_errors_in_delay):.4f}"),
        "kurtosis": float(f"{scipy.stats.kurtosis(filtered_errors_in_delay):.4f}"),
    }
    for tau in tolerances:
        if in_seconds:
            results[f"{tau}ms"] = float(
                f"{np.sum(np.abs(errors_in_delay) <= tau) / total_length:.4f}"
            )
        else:
            results[f"{tau}b"] = float(
                f"{np.sum(np.abs(errors_in_delay) <= tau) / total_length:.4f}"
            )
    results["count"] = len(filtered_abs_errors_in_delay)
    pcr_threshold = f"{tolerances[-1]}ms" if in_seconds else f"{tolerances[-1]}b"
    results["pcr"] = results[f"{pcr_threshold}"]
    return results
