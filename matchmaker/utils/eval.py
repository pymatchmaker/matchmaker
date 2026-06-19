from pathlib import Path

import numpy as np
import partitura as pt
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
    Transfer positions between score and performance using the alignment path.

    Parameters
    ----------
    wp : np.array with shape (2, T)
        Alignment path. wp[0] = performance time (seconds), wp[1] = score beats.
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

    wp_perf = wp[0].astype(float)
    wp_score = wp[1].astype(float)
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


def evaluate_alignment(
    wp_score,
    wp_perf_sec,
    gt_score_beats,
    gt_perf_sec,
    beat_tolerances=TOLERANCES_IN_BEATS,
    ms_tolerances=TOLERANCES_IN_MILLISECONDS,
):
    """Evaluate alignment quality in both beat and ms domains.

    Parameters
    ----------
    wp_score : np.ndarray
        Alignment path score axis (beats).
    wp_perf_sec : np.ndarray
        Alignment path performance axis (seconds).
    gt_score_beats : np.ndarray
        Ground truth score positions (beats).
    gt_perf_sec : np.ndarray
        Ground truth performance times (seconds).
    beat_tolerances : list
        Tolerance thresholds for beat error.
    ms_tolerances : list
        Tolerance thresholds for ms error.

    Returns
    -------
    dict
        {"beat": {...}, "ms": {...}}
    """
    # GT interpolator: score beat → perf time
    valid = np.isfinite(gt_perf_sec) & np.isfinite(gt_score_beats)
    gt_interp = scipy.interpolate.interp1d(
        gt_score_beats[valid],
        gt_perf_sec[valid],
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    # Beat metrics: perf→score prediction
    wp_sec = np.array([wp_perf_sec, wp_score])
    score_predicted = transfer_positions(
        wp_sec,
        gt_perf_sec,
        frame_rate=1,
        domain="score",
    )
    n_valid = min(len(gt_score_beats), len(score_predicted))
    beat_results = get_evaluation_results(
        gt_score_beats[:n_valid],
        score_predicted[:n_valid],
        total_counts=len(gt_score_beats),
        tolerances=beat_tolerances,
        in_seconds=False,
    )

    # Ms metrics: score→perf prediction
    gt_perf_for_wp = gt_interp(wp_score)
    ms_results = get_evaluation_results(
        gt_perf_for_wp,
        wp_perf_sec,
        total_counts=len(wp_score),
        tolerances=ms_tolerances,
    )

    return {"beat": beat_results, "ms": ms_results}


def gt_from_match(match_path, score_notes):
    perf, alignment, match_score = pt.load_match(str(match_path), create_score=True)
    match_notes = match_score.note_array()
    shared_ids = {str(n["id"]) for n in score_notes} & {
        str(n["id"]) for n in match_notes
    }
    if shared_ids:
        beat_at = {str(n["id"]): float(n["onset_beat"]) for n in score_notes}
        beat_of_match = {str(n["id"]): beat_at.get(str(n["id"])) for n in match_notes}
    else:
        beat_at = {
            (int(n["pitch"]), round(float(n["onset_beat"]), 4)): float(n["onset_beat"])
            for n in score_notes
        }
        beat_of_match = {
            str(n["id"]): beat_at.get(
                (int(n["pitch"]), round(float(n["onset_beat"]), 4))
            )
            for n in match_notes
        }
    onset_at = {str(n["id"]): float(n["onset_sec"]) for n in perf.note_array()}

    beats, secs = [], []
    for a in alignment:
        if a.get("label") != "match" or a["performance_id"] not in onset_at:
            continue
        beat = beat_of_match.get(str(a["score_id"]))
        if beat is not None:
            beats.append(beat)
            secs.append(onset_at[a["performance_id"]])

    beats = np.array(beats, dtype=float)
    secs = np.array(secs, dtype=float)
    order = np.argsort(secs, kind="stable")
    beats, secs = beats[order], secs[order]
    keep = np.ones(len(beats), dtype=bool)
    keep[1:] = beats[1:] != beats[:-1]
    return secs[keep], beats[keep]


def resolve_gt(gt, score_notes):
    """Return GT as (perf_sec, score_beat) arrays, from an ndarray, .match, or .tsv."""
    if isinstance(gt, np.ndarray):
        # gt array columns: [perf_sec, score_beat]
        arr = np.asarray(gt, dtype=float)
        return arr[:, 0], arr[:, 1]
    if Path(gt).suffix.lower() == ".match":
        return gt_from_match(gt, score_notes)
    with open(gt) as f:
        header = f.readline().rstrip("\n").split("\t")
    data = np.loadtxt(gt, delimiter="\t", skiprows=1, ndmin=2)
    return data[:, header.index("perf_sec")], data[:, header.index("score_beat")]
