from pathlib import Path
from typing import Optional, Union

import numpy as np
import partitura as pt
import scipy

TOLERANCES_IN_MILLISECONDS = [50, 100, 300, 500, 1000, 2000]
TOLERANCES_IN_BEATS = [0.05, 0.1, 0.3, 0.5, 1, 2]
ONSET_WINDOW_DURATION = 30.0


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


def evaluate_beat_errors(
    gt_score_beats,
    predicted_score_beats,
    beat_tolerances=TOLERANCES_IN_BEATS,
    *,
    total_counts=None,
):
    """Summarize beat errors and tolerance accuracy, including pooled events."""
    beat = get_evaluation_results(
        gt_score_beats,
        predicted_score_beats,
        total_counts=len(gt_score_beats) if total_counts is None else total_counts,
        tolerances=beat_tolerances,
        in_seconds=False,
    )
    return {
        "mean_b": beat["mean"],
        "median_b": beat["median"],
        "std_b": beat["std"],
        "accuracy_b": {
            threshold: accuracy
            for threshold, accuracy in beat.items()
            if threshold.endswith("b")
        },
        "skewness": beat["skewness"],
        "kurtosis": beat["kurtosis"],
    }


def onset_wise_windows(
    perf_onsets: np.ndarray, window_duration: float = ONSET_WINDOW_DURATION
) -> list[dict]:
    """Forward windows anchored at unique, finite GT onsets.

    Indices address the stably sorted finite onsets, including simultaneous
    notes. Stop after the first window that reaches the final onset; a short
    performance therefore has one whole-performance window.
    """
    if not np.isfinite(window_duration) or window_duration <= 0:
        raise ValueError("window_duration must be finite and positive")
    times = np.asarray(perf_onsets, dtype=float)
    times = np.sort(times[np.isfinite(times)], kind="stable")
    if len(times) == 0:
        return []
    final_onset = float(times[-1])
    unique_starts = np.flatnonzero(np.r_[True, times[1:] != times[:-1]])
    windows = []
    for left in unique_starts:
        start = float(times[left])
        nominal_stop = start + window_duration
        right = int(np.searchsorted(times, nominal_stop, side="right"))
        reaches_final_onset = right == len(times)
        windows.append(
            {
                "t0": start,
                "t1": min(nominal_stop, final_onset),
                "onset_index_start": int(left),
                "onset_index_stop": right,
                "n_points": right - int(left),
                "n_onsets": int(len(np.unique(times[left:right]))),
                "short_piece_fallback": left == 0 and reaches_final_onset,
                "reaches_final_onset": reaches_final_onset,
            }
        )
        if reaches_final_onset:
            break
    return windows


def spectral_arc_length(
    speed: np.ndarray,
    fs: float,
    padlevel: int = 4,
    fc: float = 10.0,
    amp_th: float = 0.05,
) -> Optional[float]:
    """Adaptive SPARC of a speed magnitude profile (Balasubramanian, 2015).

    Return None for stationary/insufficient profiles or an undefined band.
    """
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be finite and positive")
    if not np.isfinite(fc) or fc <= 0:
        raise ValueError("fc must be finite and positive")
    if not np.isfinite(amp_th) or not 0 < amp_th <= 1:
        raise ValueError("amp_th must be in (0, 1]")
    if not isinstance(padlevel, (int, np.integer)) or padlevel < 0:
        raise ValueError("padlevel must be a nonnegative integer")
    speed = np.asarray(speed, dtype=float)
    if len(speed) < 2 or not np.all(np.isfinite(speed)) or not np.any(speed):
        return None
    n_fft = int(2 ** (np.ceil(np.log2(len(speed))) + padlevel))
    spectrum = np.abs(np.fft.rfft(speed, n=n_fft))
    spectrum /= spectrum.max()
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    support = np.flatnonzero((frequencies <= fc) & (spectrum >= amp_th))
    if len(support) < 2:
        return None
    band = slice(support[0], support[-1] + 1)
    frequencies, spectrum = frequencies[band], spectrum[band]
    span = frequencies[-1] - frequencies[0]
    return float(-np.hypot(np.diff(frequencies) / span, np.diff(spectrum)).sum())


def compute_sparc(
    score_beat: np.ndarray,
    perf_sec: np.ndarray,
    fs: float = 50.0,
    fc: float = 10.0,
    padlevel: int = 4,
    *,
    onset_times: Optional[np.ndarray] = None,
    window_sec: float = ONSET_WINDOW_DURATION,
    amp_th: float = 0.05,
    return_details: bool = False,
) -> Union[float, dict, None]:
    """Mean adaptive SPARC over GT onset-wise windows, without time weights.

    Each window uses the full position trajectory resampled at ``fs``, holding
    the last reported position between updates, rather than sampling positions
    only at onsets. Stationary/undefined windows are omitted from the mean; an
    entirely undefined path returns None.

    Pass the same GT ``onset_times`` for every follower being compared, as
    ``evaluate_alignment`` does. Standalone calls without GT use the path's
    timestamps as anchors. With ``return_details=True``, return the scalar,
    total/valid window counts, valid fraction and per-window diagnostics.
    """
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be finite and positive")
    score_beat = np.asarray(score_beat, dtype=float)
    perf_sec = np.asarray(perf_sec, dtype=float)
    if score_beat.ndim != 1 or perf_sec.ndim != 1 or score_beat.shape != perf_sec.shape:
        raise ValueError(
            "score_beat and perf_sec must be matching one-dimensional arrays"
        )
    valid = np.isfinite(score_beat) & np.isfinite(perf_sec)
    score_beat, perf_sec = score_beat[valid], perf_sec[valid]
    order = np.argsort(perf_sec, kind="stable")
    score_beat, perf_sec = score_beat[order], perf_sec[order]
    # Keep the final decision at duplicate timestamps, as in transfer_positions.
    last_at_time = np.r_[perf_sec[1:] != perf_sec[:-1], True] if len(perf_sec) else []
    score_beat, perf_sec = score_beat[last_at_time], perf_sec[last_at_time]
    windows = onset_wise_windows(
        perf_sec if onset_times is None else onset_times, window_sec
    )
    diagnostics = []
    values = []
    cached_windows = {}
    if windows and len(perf_sec) >= 2:
        times = np.arange(windows[0]["t0"], windows[-1]["t1"], 1.0 / fs)
        report_indices = np.searchsorted(perf_sec, times, side="right") - 1
        positions = np.full(len(times), np.nan)
        has_report = report_indices >= 0
        positions[has_report] = score_beat[report_indices[has_report]]
    else:
        times, positions = np.array([]), np.array([])
    for window in windows:
        start, stop = window["t0"], window["t1"]
        value, reason = None, "insufficient_samples"
        if len(perf_sec) >= 2 and (start < perf_sec[0] or stop > perf_sec[-1]):
            reason = "uncovered"
        elif len(times):
            left, right = np.searchsorted(times, [start, stop], side="left")
            key = (int(left), int(right))
            if key not in cached_windows:
                position = positions[left:right]
                if len(position) >= 2:
                    speed = np.abs(np.gradient(position, 1.0 / fs))
                    value = spectral_arc_length(speed, fs, padlevel, fc, amp_th)
                    reason = "stationary" if not np.any(speed) else "undefined_band"
                    if value is not None:
                        reason = "ok"
                cached_windows[key] = value, reason
            value, reason = cached_windows[key]
        diagnostics.append({"t0": start, "t1": stop, "sparc": value, "reason": reason})
        if value is not None:
            values.append(value)
    result = {
        "sparc": float(np.mean(values)) if values else None,
        "sparc_n_windows": len(windows),
        "sparc_n_valid_windows": len(values),
        "sparc_valid_window_fraction": len(values) / len(windows) if windows else None,
        "sparc_windows": diagnostics,
    }
    return result if return_details else result["sparc"]


def evaluate_alignment(
    aligned_score_beats,
    aligned_perf_sec,
    gt_score_beats,
    gt_perf_sec,
    beat_tolerances=TOLERANCES_IN_BEATS,
    ms_tolerances=TOLERANCES_IN_MILLISECONDS,
    sparc_window_sec=ONSET_WINDOW_DURATION,
    *,
    return_details: bool = False,
):
    """Evaluate beat accuracy and trajectory smoothness for one performance.

    Parameters
    ----------
    aligned_score_beats : np.ndarray
        Alignment path score axis (beats).
    aligned_perf_sec : np.ndarray
        Alignment path performance axis (seconds).
    gt_score_beats : np.ndarray
        Ground truth score positions (beats).
    gt_perf_sec : np.ndarray
        Ground truth performance times (seconds).
    beat_tolerances : list
        Tolerance thresholds for beat error.
    ms_tolerances : list
        Tolerance thresholds for ms error, used when ``return_details=True``.
    sparc_window_sec : float
        Forward-window duration for GT onset-wise SPARC (default: 30 seconds).
    return_details : bool
        Include ms error metrics and full-precision SPARC diagnostics for
        benchmark aggregation. Default: False.

    Returns
    -------
    dict
        ``mean_b``, ``median_b``, ``std_b``, ``accuracy_b``, ``skewness``,
        ``kurtosis``, and ``smoothness``. With ``return_details=True``, also
        includes ``ms`` metrics and a ``smoothness_details`` dictionary.
    """
    valid = np.isfinite(gt_perf_sec) & np.isfinite(gt_score_beats)

    # Beat metrics: perf→score prediction
    wp_sec = np.array([aligned_perf_sec, aligned_score_beats])
    score_predicted = transfer_positions(
        wp_sec,
        gt_perf_sec,
        frame_rate=1,
        domain="score",
    )
    n_valid = min(len(gt_score_beats), len(score_predicted))
    results = evaluate_beat_errors(
        gt_score_beats[:n_valid],
        score_predicted[:n_valid],
        total_counts=len(gt_score_beats),
        beat_tolerances=beat_tolerances,
    )

    sparc_results = compute_sparc(
        aligned_score_beats,
        aligned_perf_sec,
        onset_times=np.asarray(gt_perf_sec)[valid],
        window_sec=sparc_window_sec,
        return_details=True,
    )

    smoothness = sparc_results["sparc"]
    results["smoothness"] = round(smoothness, 4) if smoothness is not None else None
    if return_details:
        gt_interp = scipy.interpolate.interp1d(
            gt_score_beats[valid],
            gt_perf_sec[valid],
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        results["ms"] = get_evaluation_results(
            gt_interp(aligned_score_beats),
            aligned_perf_sec,
            total_counts=len(aligned_score_beats),
            tolerances=ms_tolerances,
        )
        results["smoothness_details"] = sparc_results
    return results


def _streamed_onsets(performance, match_perf, performance_ids):
    """Onset times of ``performance`` by note id, if it holds the match's notes.

    A .match file and the MIDI file a follower streams can store the same notes
    with different tick resolutions (e.g. (n)ASAP: 480 vs 384 ppq), so their
    times differ by up to half a tick. The follower stamps its updates with the
    times it receives, so GT must use them: a GT time a fraction of a
    millisecond early would be scored against the position before that update.
    Returns None unless every matched note has the same id and pitch in both.
    """
    if performance is None:
        return None
    if isinstance(performance, (str, Path)):
        performance = pt.load_performance(str(performance))
    parts = getattr(performance, "performedparts", [performance])
    streamed = {str(n["id"]): n for part in parts for n in part.notes}
    pitch_in_match = {
        str(n["id"]): int(n["pitch"])
        for part in match_perf.performedparts
        for n in part.notes
    }
    for pid in performance_ids:
        note = streamed.get(pid)
        if note is None or int(note["pitch"]) != pitch_in_match[pid]:
            return None
    return {pid: float(note["note_on"]) for pid, note in streamed.items()}


def gt_from_match(match_path, score_notes, performance=None):
    """GT (perf_sec, score_beat) from a .match file.

    ``performance`` is the performance the follower receives (a MIDI path or a
    partitura performance); when its notes are those of the match file, their
    onset times are used instead of the match file's own.
    """
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
    onset_at = {
        str(n["id"]): float(n["note_on"])
        for part in perf.performedparts
        for n in part.notes
    }
    matched_ids = [
        str(a["performance_id"])
        for a in alignment
        if a.get("label") == "match" and str(a["performance_id"]) in onset_at
    ]
    streamed = _streamed_onsets(performance, perf, matched_ids)
    if streamed is not None:
        onset_at = streamed

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


def resolve_gt(gt, score_notes, performance=None):
    """Return GT as (perf_sec, score_beat) arrays, from an ndarray, .match, or .tsv.

    ``performance``: the performance the follower receives; see gt_from_match.
    """
    if isinstance(gt, np.ndarray):
        # gt array columns: [perf_sec, score_beat]
        arr = np.asarray(gt, dtype=float)
        return arr[:, 0], arr[:, 1]
    if Path(gt).suffix.lower() == ".match":
        return gt_from_match(gt, score_notes, performance=performance)
    with open(gt) as f:
        header = f.readline().rstrip("\n").split("\t")
    data = np.loadtxt(gt, delimiter="\t", skiprows=1, ndmin=2)
    return data[:, header.index("perf_sec")], data[:, header.index("score_beat")]
