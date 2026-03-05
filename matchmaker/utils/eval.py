from typing import Union

import numpy as np
import scipy

TOLERANCES_IN_MILLISECONDS = [50, 100, 300, 500, 1000, 2000]
TOLERANCES_IN_BEATS = [0.05, 0.1, 0.3, 0.5, 1, 2]


def _forward_fill_nan(arr):
    """Forward-fill NaN values in-place."""
    last_valid = np.nan
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            arr[i] = last_valid
        else:
            last_valid = arr[i]


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
    """Transfer reference annotations to target positions via warping path.

    Supports two warping-path conventions:
    - **beat**: wp[0] = float beat positions, wp[1] = target frame indices.
    - **state**: wp[0] = integer state indices, wp[1] = target frame indices.

    Parameters
    ----------
    wp : np.array, shape (2, T)
    ref_anns : array-like
        Beat positions (beat mode) or dummy sequence for state count (state mode).
    frame_rate : float
    reverse : bool
        If True, transfer target -> reference direction.
    mode : {"auto", "state", "beat"}
    reducer : {"min", "max", "median", "mean"}
    state_offset : {"auto"} or int
    output : {"seconds", "frames"}
    """
    if output not in {"seconds", "frames"}:
        raise ValueError(f"Invalid output={output!r}. Use 'seconds' or 'frames'.")
    if mode not in {"auto", "state", "beat"}:
        raise ValueError(f"Invalid mode={mode!r}. Use 'auto', 'state', or 'beat'.")

    if mode == "auto":
        if np.issubdtype(wp[0].dtype, np.floating):
            mode = "beat"
        else:
            mode = "state"

    if reverse:
        x, y = wp[1], wp[0]
    else:
        x, y = wp[0], wp[1]

    if mode == "beat":
        return _transfer_beat(wp, ref_anns, frame_rate, reverse, reducer, output)

    if reverse:
        return _transfer_state_reverse(x, y, ref_anns, frame_rate, output)

    return _transfer_state_forward(
        x, y, ref_anns, frame_rate, reducer, state_offset, output
    )


def _transfer_beat(wp, ref_anns, frame_rate, reverse, reducer, output):
    if reverse:
        query_frames = np.asarray(ref_anns) * frame_rate
        predicted_beats = np.interp(query_frames, wp[1], wp[0])
        predicted_beats[query_frames < wp[1][0]] = np.nan
        return predicted_beats

    reduce_fn = {"min": np.min, "max": np.max, "median": np.median, "mean": np.mean}[
        reducer
    ]
    score_beats = np.asarray(ref_anns, dtype=float)
    wp_beats, wp_frames = wp[0], wp[1]
    predicted = np.full(len(score_beats), np.nan)

    for i, sb in enumerate(score_beats):
        idx = np.where(wp_beats == sb)[0]
        if idx.size == 0:
            past_mask = wp_beats <= sb
            if not np.any(past_mask):
                continue
            idx = np.where(wp_beats == wp_beats[past_mask][-1])[0]
        predicted[i] = float(reduce_fn(wp_frames[idx]))

    _forward_fill_nan(predicted)
    return predicted if output == "frames" else predicted / frame_rate


def _transfer_state_reverse(x, y, ref_anns, frame_rate, output):
    """Causal nearest-neighbor lookup for state mode reverse."""
    ref_frames = np.round(np.asarray(ref_anns) * frame_rate)
    predicted = np.full(len(ref_frames), np.nan)

    for i, r in enumerate(ref_frames):
        past = np.where(x <= r)[0]
        if past.size > 0:
            max_x = x[past[-1]]
            predicted[i] = float(np.min(y[np.where(x == max_x)[0]]))

    return predicted if output == "frames" else predicted / frame_rate


def _transfer_state_forward(x, y, ref_anns, frame_rate, reducer, state_offset, output):
    """State-index forward transfer."""
    reduce_fn = {"min": np.min, "max": np.max, "median": np.median, "mean": np.mean}[
        reducer
    ]
    num_states = len(ref_anns)
    predicted = np.full(num_states, np.nan)

    x_int = np.asarray(x, dtype=int)
    y_int = np.asarray(y, dtype=int)

    if state_offset == "auto":
        observed = np.unique(x_int)
        candidates = list(dict.fromkeys([0, 1, int(np.min(x_int))]))
        offset = max(
            candidates,
            key=lambda off: np.intersect1d(
                observed, np.arange(off, off + num_states, dtype=int)
            ).size,
        )
    else:
        offset = int(state_offset)

    for s in range(num_states):
        idx = np.where(x_int == s + offset)[0]
        if idx.size > 0:
            predicted[s] = float(reduce_fn(y_int[idx].astype(float)))

    _forward_fill_nan(predicted)
    return predicted if output == "frames" else predicted / frame_rate


def get_evaluation_results(
    gt_annots,
    predicted_annots,
    total_counts,
    tolerances=TOLERANCES_IN_MILLISECONDS,
    in_seconds=True,
):
    if in_seconds:
        errors_in_delay = (gt_annots - predicted_annots) * 1000
    else:
        errors_in_delay = gt_annots - predicted_annots

    abs_errors_in_delay = np.abs(errors_in_delay)

    results = {
        "mean": float(f"{np.nanmean(abs_errors_in_delay):.4f}"),
        "median": float(f"{np.nanmedian(abs_errors_in_delay):.4f}"),
        "std": float(f"{np.nanstd(abs_errors_in_delay):.4f}"),
        "skewness": float(f"{scipy.stats.skew(errors_in_delay):.4f}"),
        "kurtosis": float(f"{scipy.stats.kurtosis(errors_in_delay):.4f}"),
    }

    if in_seconds:
        for tau in tolerances:
            results[f"{tau}ms"] = float(
                f"{np.sum(abs_errors_in_delay <= tau) / total_counts:.4f}"
            )
    else:
        for tau in tolerances:
            results[f"{tau}b"] = float(
                f"{np.sum(abs_errors_in_delay <= tau) / total_counts:.4f}"
            )

    pcr_key = f"{tolerances[-1]}ms" if in_seconds else f"{tolerances[-1]}b"
    results["pcr"] = results[pcr_key]
    results["count"] = len(abs_errors_in_delay)
    return results
