import numpy as np
import scipy

TOLERANCES_IN_MILLISECONDS = [50, 100, 300, 500, 1000, 2000]
TOLERANCES_IN_BEATS = [0.05, 0.1, 0.3, 0.5, 1, 2]


def transfer_positions(
    wp,
    ref_anns,
    frame_rate,
    *,
    domain: str = "score",
    aggregation_func=None,
):
    """
    Transfer positions between score and performance using the warping path.

    Parameters
    ----------
    wp : np.array with shape (2, T)
        Warping path. wp[0] = score beats, wp[1] = performance frame indices.
    ref_anns : array-like
        Query positions (seconds for domain="score",
        beats for domain="performance").
    frame_rate : int
        Frame rate of the audio.
    domain : {"score", "performance"}
        Domain of the output.
        "score": perf→score lookup. Given performance times (seconds),
            return predicted score positions (beats).
        "performance": score→perf lookup. Given score beats, return
            predicted performance times (seconds).
    aggregation_func : callable or None
        Function to aggregate multiple values sharing the same key
        (e.g., np.max, np.min, np.mean). If None, defaults to:
        - domain="score": last entry in temporal order (tracker's
          final decision at that frame)
        - domain="performance": np.min (earliest arrival at that beat,
          i.e. first-crossing rule)

    Returns
    -------
    predicted : np.array
        Predicted positions in the target domain.
    """
    if domain not in {"score", "performance"}:
        raise ValueError(f"Invalid domain={domain!r}. Use 'score' or 'performance'.")

    wp_score = wp[0].astype(float)
    wp_perf = wp[1].astype(float)
    queries = np.asarray(ref_anns, dtype=float)

    def _last(arr):
        return arr[-1]

    if aggregation_func is None:
        aggregation_func = _last if domain == "score" else np.min

    if domain == "score":
        # Perf → Score: "at perf time t, what is the tracker's score position?"
        # Group by perf frame, take the last entry by default (tracker's final decision).
        query_frames = queries * frame_rate

        sort_idx = np.argsort(wp_perf, kind="stable")
        wp_perf_sorted = wp_perf[sort_idx]
        wp_score_sorted = wp_score[sort_idx]

        unique_frames, first_idx = np.unique(wp_perf_sorted, return_index=True)
        reduced_scores = np.empty(len(unique_frames))
        for g in range(len(unique_frames)):
            start = first_idx[g]
            end = (
                first_idx[g + 1] if g + 1 < len(unique_frames) else len(wp_score_sorted)
            )
            reduced_scores[g] = aggregation_func(wp_score_sorted[start:end])

        # unique_frames is monotonic → searchsorted for last frame ≤ query
        indices = np.searchsorted(unique_frames, query_frames, side="right") - 1
        predicted = np.full(len(queries), np.nan)
        valid = indices >= 0
        predicted[valid] = reduced_scores[indices[valid]]
        return predicted
    else:
        # Score → Perf: "when did the tracker first reach beat b?"
        # Group by score position, aggregate perf frame values per group.
        sort_idx = np.argsort(wp_score, kind="stable")
        wp_score_sorted = wp_score[sort_idx]
        wp_perf_sorted = wp_perf[sort_idx]

        unique_beats, first_idx = np.unique(wp_score_sorted, return_index=True)
        reduced_perf = np.empty(len(unique_beats))
        for g in range(len(unique_beats)):
            start = first_idx[g]
            end = first_idx[g + 1] if g + 1 < len(unique_beats) else len(wp_perf_sorted)
            reduced_perf[g] = aggregation_func(wp_perf_sorted[start:end])

        indices = np.searchsorted(unique_beats, queries, side="left")
        predicted = np.full(len(queries), np.nan)
        valid = indices < len(unique_beats)
        predicted[valid] = reduced_perf[indices[valid]]
        return predicted / frame_rate


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
        "skewness": float(
            f"{scipy.stats.skew(errors_in_delay, nan_policy='omit'):.4f}"
        ),
        "kurtosis": float(
            f"{scipy.stats.kurtosis(errors_in_delay, nan_policy='omit'):.4f}"
        ),
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

    return results
