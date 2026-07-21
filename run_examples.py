import argparse
import csv
import datetime
import json
from _queue import Empty
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from matchmaker import EXAMPLE_PIECES, Matchmaker
from matchmaker.utils.eval import evaluate_alignment, resolve_gt

ROOT_DIR = Path(__file__).parent
_piece = EXAMPLE_PIECES["simple_mozart"]  # simple_mozart, bach_fugue
SCORE_FILE = Path(_piece["score"])
PERFORMANCE_AUDIO_FILE = Path(_piece["audio"])
PERFORMANCE_MIDI_FILE = Path(_piece["midi"])
MATCH_FILE = Path(_piece["match"])


def select_performance_file(input_mode):
    performance_file = (
        PERFORMANCE_MIDI_FILE if input_mode == "midi" else PERFORMANCE_AUDIO_FILE
    )
    print(f"Performance file: {performance_file.name}, with input mode: {input_mode}")
    return performance_file


def evaluate(mm, match_file):
    """Evaluate a completed run against a .match ground truth."""
    wp = mm.score_follower.alignment_path
    perf_sec = mm._wp_perf_to_seconds(wp[0].astype(float))
    score_beat = wp[1].astype(float)

    gt_perf, gt_score = resolve_gt(str(match_file), mm.score_part.note_array())
    results = evaluate_alignment(score_beat, perf_sec, gt_score, gt_perf)

    if mm.alignment_duration is not None:
        finite_perf = gt_perf[np.isfinite(gt_perf)]
        perf_dur = float(finite_perf.max() - finite_perf.min())
        if perf_dur > 0:
            results["rtf"] = float(f"{mm.alignment_duration / perf_dur:.4f}")
    if mm.input_type == "audio":
        results.update(mm.get_latency_stats())

    return results, perf_sec, score_beat, gt_perf, gt_score


def save_tsv(rows, path):
    with open(path, "w") as f:
        f.write("perf_sec\tscore_beat\n")
        csv.writer(f, delimiter="\t").writerows(rows)


def plot_alignment_path(perf_sec, score_beat, gt_perf, gt_score, save_path, run_name):
    """Plot alignment path, ground truth, and predicted score positions."""
    pred_score = np.interp(gt_perf, perf_sec, score_beat)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        perf_sec,
        score_beat,
        s=6,
        color="limegreen",
        alpha=0.85,
        label="full alignment path",
        zorder=3,
    )
    ax.scatter(
        gt_perf,
        pred_score,
        s=6,
        color="royalblue",
        label="predicted @ GT onsets",
        zorder=4,
    )
    ax.scatter(
        gt_perf, gt_score, s=12, marker="x", color="red", label="ground truth", zorder=5
    )
    ax.set_xlabel("performance time (s)")
    ax.set_ylabel("score position (beats)")
    ax.set_title(f"alignment ({run_name})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_results(save_dir, run_name, results, perf_sec, score_beat, gt_perf, gt_score):
    """Save wp/gt TSVs, results JSON, and the alignment path plot."""
    save_dir.mkdir(parents=True, exist_ok=True)
    save_tsv(np.column_stack([perf_sec, score_beat]), save_dir / f"wp_{run_name}.tsv")
    save_tsv(np.column_stack([gt_perf, gt_score]), save_dir / f"gt_{run_name}.tsv")
    with open(save_dir / f"{run_name}.json", "w") as f:
        json.dump(results, f, indent=4)
    plot_alignment_path(
        perf_sec, score_beat, gt_perf, gt_score, save_dir / f"{run_name}.png", run_name
    )


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Matchmaker in simulation mode")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--audio", action="store_true", help="Use audio input mode")
    group.add_argument("--midi", action="store_true", help="Use MIDI input mode")
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Score following method (e.g., arzt, dixon, outerhmm, audio_outerhmm)",
    )
    args = parser.parse_args()

    input_mode = "midi" if args.midi else "audio"
    performance_file = select_performance_file(input_mode)

    print(f"Running matchmaker with the score file ({SCORE_FILE.name})...")
    print("-" * 50)

    if args.method is not None:
        method = args.method
    else:
        method = "pthmm" if input_mode == "midi" else "arzt"

    # Initialize matchmaker (simulation mode)
    try:
        mm = Matchmaker(
            score_file=SCORE_FILE,
            performance_file=performance_file,
            input_type=input_mode,
            method=method,
        )
    except Empty as e:
        print(f"Error initializing Matchmaker: {e}")
        return

    # Run real-time score following
    for current_position in mm.run():
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] Current beat position: {current_position}")

    print("-" * 50)
    print(f"Running evaluation using the match file ({MATCH_FILE.name})...")

    results, perf_sec, score_beat, gt_perf, gt_score = evaluate(mm, MATCH_FILE)
    print(f"Evaluation Result: {json.dumps(results, indent=4)}")

    results_dir = ROOT_DIR / "results"
    save_results(
        results_dir, "simple_example", results, perf_sec, score_beat, gt_perf, gt_score
    )
    print(f"Detailed evaluation results saved in {results_dir}")


if __name__ == "__main__":
    main()
