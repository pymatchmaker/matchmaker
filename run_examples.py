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


def save_results(save_dir, run_name, results, perf_sec, score_beat, gt_perf, gt_score, save_plots=True):
    """Save alignment data, metrics, and an optional plot."""
    save_dir.mkdir(parents=True, exist_ok=True)
    save_tsv(np.column_stack([perf_sec, score_beat]), save_dir / f"wp_{run_name}.tsv")
    save_tsv(np.column_stack([gt_perf, gt_score]), save_dir / f"gt_{run_name}.tsv")
    with open(save_dir / f"{run_name}.json", "w") as f:
        json.dump(results, f, indent=4)
    if save_plots:
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
        "--piece",
        type=str,
        default="simple_mozart",
        choices=list(EXAMPLE_PIECES.keys()),
        help="Built-in example piece to run (default: simple_mozart)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Score following method (e.g., arzt, arzt_tempo, dixon, outerhmm)",
    )
    parser.add_argument("--score", type=str, default=None, help="Path to custom score XML/MusicXML")
    parser.add_argument("--audio-file", type=str, default=None, help="Path to custom performance audio file")
    parser.add_argument("--midi-file", type=str, default=None, help="Path to custom performance MIDI file")
    parser.add_argument("--match", type=str, default=None, help="Path to custom .match ground truth file")
    parser.add_argument("--unfold", action="store_true", help="Unfold score repetitions during load")
    parser.add_argument("--no-plots", action="store_true", help="Skip alignment plots")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "results", help="Result directory")
    parser.add_argument("--kwargs", type=json.loads, default={}, help="Method option overrides")
    args = parser.parse_args()

    input_mode = "midi" if args.midi else "audio"

    # Resolve piece files
    if args.score:
        score_file = Path(args.score)
        performance_file = Path(args.midi_file if input_mode == "midi" else args.audio_file)
        match_file = Path(args.match) if args.match else None
        run_name = score_file.stem
    else:
        piece_cfg = EXAMPLE_PIECES[args.piece]
        score_file = Path(piece_cfg["score"])
        performance_file = Path(piece_cfg["midi"] if input_mode == "midi" else piece_cfg["audio"])
        match_file = Path(piece_cfg["match"]) if "match" in piece_cfg else None
        run_name = args.piece

    print(f"Performance file: {performance_file.name}, with input mode: {input_mode}")
    print(f"Running matchmaker with the score file ({score_file.name})...")
    print("-" * 50)

    if args.method is not None:
        method = args.method
    else:
        method = "pthmm" if input_mode == "midi" else "arzt"

    # Initialize matchmaker (simulation mode)
    try:
        mm = Matchmaker(
            score_file=score_file,
            performance_file=performance_file,
            input_type=input_mode,
            method=method,
            unfold_score=args.unfold,
            kwargs=args.kwargs or None,
        )
    except Empty as e:
        print(f"Error initializing Matchmaker: {e}")
        return

    # Run real-time score following
    for current_position in mm.run():
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] Current beat position: {current_position}")

    if match_file and match_file.is_file():
        print("-" * 50)
        print(f"Running evaluation using the match file ({match_file.name})...")

        results, perf_sec, score_beat, gt_perf, gt_score = evaluate(mm, match_file)
        print(f"Evaluation Result: {json.dumps(results, indent=4)}")

        results_dir = args.output_dir
        save_results(
            results_dir, f"{run_name}_{method}", results, perf_sec, score_beat, gt_perf, gt_score,
            save_plots=not args.no_plots,
        )
        print(f"Detailed evaluation results saved in {results_dir}")
    else:
        print("\nAlignment finished (no match file provided for quantitative evaluation).")
    return


if __name__ == "__main__":
    main()
