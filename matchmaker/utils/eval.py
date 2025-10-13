from typing import TypedDict, Union

import numpy as np
import scipy

TOLERANCES_IN_MILLISECONDS = [50, 100, 300, 500, 1000, 2000]
TOLERANCES_IN_BEATS = [0.05, 0.1, 0.3, 0.5, 1, 2]


def transfer_positions(
    wp,
    ref_anns,
    frame_rate,
    reverse=False,
    *,
    mode: str = "auto",
    reducer: str = "min",
    state_offset: Union[int, str] = "auto",
    output: str = "seconds",
):
    """
    Transfer the positions of the reference annotations to the target annotations using the warping path.

    This function supports two common warping-path conventions:

    - **frame mode** (classic DTW-style): wp[0] and wp[1] are frame indices for reference/target features.
    - **state mode** (HMM/score-state): wp[0] contains *reference state indices* and wp[1] contains *target frame indices*.

    Parameters
    ----------
    wp : np.array with shape (2, T)
        array of warping path.
        warping_path[0] is the index of the reference (score) feature and warping_path[1] is the index of the target(input) feature.
    ref_ann : List[float]
        In **frame mode**, reference annotations in seconds.
        In **state mode**, a sequence whose length equals the number of reference states (e.g., score unique_onsets);
        the values are not used except for determining the number of states.
    frame_rate : int
        frame rate of the audio.
    reverse : bool
        If True, swap the direction (target -> reference).
    mode : {"auto", "frame", "state"}
        Warping-path convention. "auto" picks "state" when wp[0] looks like small discrete state indices.
    reducer : {"min", "max", "median", "mean"}
        In **state mode**, how to select a single representative target frame for each state when multiple wp entries
        map to the same state.
    state_offset : {"auto"} or int
        In **state mode**, wp[0] may start at 0 or 1 (or have a leading start-state). "auto" chooses the offset that
        best matches the expected number of states.
    output : {"seconds", "frames"}
        Return unit. "seconds" divides frames by frame_rate; "frames" returns frame indices.

    Returns
    -------
    predicted_targets : np.array with shape (T,)
        Predicted target positions (seconds or frames depending on output).
    """
    if output not in {"seconds", "frames"}:
        raise ValueError(f"Invalid output={output!r}. Use 'seconds' or 'frames'.")

    if reverse:
        x, y = wp[1], wp[0]
    else:
        x, y = wp[0], wp[1]

    if mode not in {"auto", "frame", "state"}:
        raise ValueError(f"Invalid mode={mode!r}. Use 'auto', 'frame', or 'state'.")

    # Heuristic: state paths have small discrete indices (often << target frames),
    # while frame paths typically cover most reference frames (unique count is large).
    if mode == "auto":
        x_unique = np.unique(x)
        n_ref = len(ref_anns)
        looks_like_state = (x_unique.size <= max(4, 2 * n_ref)) and (
            int(np.max(x)) <= max(10, 5 * n_ref)
        )
        mode = "state" if looks_like_state else "frame"

    if mode == "frame":
        # Causal nearest neighbor interpolation (reference seconds -> reference frames -> target frames)
        ref_anns_frame = np.round(np.asarray(ref_anns) * frame_rate)
        predicted_targets = np.ones(len(ref_anns_frame), dtype=float) * np.nan

        for i, r in enumerate(ref_anns_frame):
            # 1) Scan all x values less than or equal to r and find the largest x value
            past_indices = np.where(x <= r)[0]
            if past_indices.size > 0:
                # Find indices corresponding to the largest x value
                max_x_val = x[past_indices[-1]]
                max_x_indices = np.where(x == max_x_val)[0]

                # 2) Among all y values mapped to this x value, select the minimum y value
                corresponding_y_values = y[max_x_indices]
                predicted_targets[i] = float(np.min(corresponding_y_values))

        if output == "frames":
            return predicted_targets
        return np.asarray(predicted_targets) / frame_rate

    # mode == "state"
    # Goal: for each reference state index, select representative target frame from wp.
    num_states = len(ref_anns)
    predicted_frames = np.ones(num_states, dtype=float) * np.nan

    x_int = np.asarray(x, dtype=int)
    y_int = np.asarray(y, dtype=int)

    if reducer not in {"min", "max", "median", "mean"}:
        raise ValueError(
            f"Invalid reducer={reducer!r}. Use 'min', 'max', 'median', or 'mean'."
        )

    if state_offset == "auto":
        # Choose offset that maximizes overlap between expected states and observed wp state indices.
        observed = np.unique(x_int)
        candidates = []
        for off in (0, 1, int(np.min(x_int))):
            if off not in candidates:
                candidates.append(off)
        best_off = candidates[0]
        best_overlap = -1
        for off in candidates:
            expected = np.arange(off, off + num_states, dtype=int)
            overlap = np.intersect1d(observed, expected).size
            if overlap > best_overlap:
                best_overlap = overlap
                best_off = off
        offset = best_off
    else:
        offset = int(state_offset)

    for s in range(num_states):
        wp_state = s + offset
        idx = np.where(x_int == wp_state)[0]
        if idx.size == 0:
            continue
        vals = y_int[idx].astype(float)
        if reducer == "min":
            predicted_frames[s] = float(np.min(vals))
        elif reducer == "max":
            predicted_frames[s] = float(np.max(vals))
        elif reducer == "median":
            predicted_frames[s] = float(np.median(vals))
        else:  # mean
            predicted_frames[s] = float(np.mean(vals))

    if output == "frames":
        return predicted_frames
    return predicted_frames / frame_rate


def transfer_from_score_to_predicted_perf(wp, score_annots, frame_rate, mode="auto"):
    predicted_perf_idx = transfer_positions(
        wp,
        score_annots,
        frame_rate,
        mode=mode,
    )
    return predicted_perf_idx


def transfer_from_perf_to_predicted_score(wp, perf_annots, frame_rate, mode="auto"):
    predicted_score_idx = transfer_positions(
        wp, perf_annots, frame_rate, reverse=True, mode=mode
    )
    return predicted_score_idx


def get_evaluation_results(
    gt_annots,
    predicted_annots,
    total_counts,
    tolerances=TOLERANCES_IN_MILLISECONDS,
    pcr_threshold=2_000,  # 2 seconds
    in_seconds=True,
):
    if in_seconds:
        errors_in_delay = (gt_annots - predicted_annots) * 1000  # in milliseconds
    else:
        errors_in_delay = gt_annots - predicted_annots

    filtered_errors_in_delay = errors_in_delay[np.abs(errors_in_delay) <= pcr_threshold]
    filtered_abs_errors_in_delay = np.abs(filtered_errors_in_delay)

    results = {
        "mean": float(f"{np.nanmean(filtered_abs_errors_in_delay):.4f}"),
        "median": float(f"{np.nanmedian(filtered_abs_errors_in_delay):.4f}"),
        "std": float(f"{np.nanstd(filtered_abs_errors_in_delay):.4f}"),
        "skewness": float(f"{scipy.stats.skew(filtered_errors_in_delay):.4f}"),
        "kurtosis": float(f"{scipy.stats.kurtosis(filtered_errors_in_delay):.4f}"),
    }

    if in_seconds:
        for tau in tolerances:
            results[f"{tau}ms"] = float(
                f"{np.sum(np.abs(errors_in_delay) <= tau) / total_counts:.4f}"
            )
    else:
        for tau in tolerances:
            results[f"{tau}b"] = float(
                f"{np.sum(np.abs(errors_in_delay) <= tau) / total_counts:.4f}"
            )

    results["pcr"] = float(f"{len(filtered_errors_in_delay) / total_counts:.4f}")
    results["count"] = len(filtered_abs_errors_in_delay)
    return results


# def get_evaluation_results(
#     gt_annots,
#     predicted_annots,
#     total_length,
#     tolerances=TOLERANCES_IN_MILLISECONDS,
#     in_seconds=True,
# ):
#     if in_seconds:
#         errors_in_delay = (gt_annots - predicted_annots) * 1000  # in milliseconds
#     else:
#         errors_in_delay = gt_annots - predicted_annots

#     filtered_errors_in_delay = errors_in_delay[
#         np.abs(errors_in_delay) <= tolerances[-1]
#     ]
#     filtered_abs_errors_in_delay = np.abs(filtered_errors_in_delay)

#     results = {
#         "mean": float(f"{np.nanmean(filtered_abs_errors_in_delay):.4f}"),
#         "median": float(f"{np.nanmedian(filtered_abs_errors_in_delay):.4f}"),
#         "std": float(f"{np.nanstd(filtered_abs_errors_in_delay):.4f}"),
#         "skewness": float(f"{scipy.stats.skew(filtered_errors_in_delay):.4f}"),
#         "kurtosis": float(f"{scipy.stats.kurtosis(filtered_errors_in_delay):.4f}"),
#     }
#     for tau in tolerances:
#         if in_seconds:
#             results[f"{tau}ms"] = float(
#                 f"{np.sum(np.abs(errors_in_delay) <= tau) / total_length:.4f}"
#             )
#         else:
#             results[f"{tau}b"] = float(
#                 f"{np.sum(np.abs(errors_in_delay) <= tau) / total_length:.4f}"
#             )
#     results["count"] = len(filtered_abs_errors_in_delay)
#     pcr_threshold = f"{tolerances[-1]}ms" if in_seconds else f"{tolerances[-1]}b"
#     results["pcr"] = results[f"{pcr_threshold}"]
#     return results
