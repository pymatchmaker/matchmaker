import argparse
import datetime
import json
from _queue import Empty
from pathlib import Path

from matchmaker import Matchmaker

ROOT_DIR = Path(__file__).parent
SCORE_FILE = ROOT_DIR / "matchmaker/assets/simple_mozart_k265_var1.musicxml"
PERFORMANCE_AUDIO_FILE = ROOT_DIR / "matchmaker/assets/simple_mozart_k265_var1.mp3"
PERFORMANCE_MIDI_FILE = ROOT_DIR / "matchmaker/assets/simple_mozart_k265_var1.mid"
ANNOTATION_FILE = (
    ROOT_DIR / "matchmaker/assets/simple_mozart_k265_var1_note_annotations.txt"
)


def select_performance_file(input_mode):
    performance_file = (
        PERFORMANCE_MIDI_FILE if input_mode == "midi" else PERFORMANCE_AUDIO_FILE
    )
    print(f"Performance file: {performance_file.name}, with input mode: {input_mode}")
    return performance_file


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run Matchmaker and evaluate the results (only in simulation mode)"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--audio", action="store_true", help="Use audio input mode")
    group.add_argument("--midi", action="store_true", help="Use MIDI input mode")
    args = parser.parse_args()

    input_mode = "midi" if args.midi else "audio"
    performance_file = select_performance_file(input_mode)

    print(f"Running matchmaker with the score file ({SCORE_FILE.name})...")
    print("-" * 50)

    # Use the corresponding score follower for each input mode.
    # - audio: AudioOuterProductHMM (paper-style HMM / outer-product)
    # - midi : OuterProductHMM
    method = "outerhmm" if input_mode == "midi" else "audio_outerhmm"
    feature_type = "pitchclass" if input_mode == "midi" else "cqt_spectral_flux"

    # Initialize matchmaker (simulation mode)
    try:
        mm = Matchmaker(
            score_file=SCORE_FILE,
            performance_file=performance_file,
            input_type=input_mode,
            method=method,
            feature_type=feature_type,
        )
    except Empty as e:
        print(f"Error initializing Matchmaker: {e}")
        return

    # Run real-time score following
    for current_position in mm.run(wait=True):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] Current beat position: {current_position}")

    # Run evaluation
    print("-" * 50)
    print(f"Running evaluation using the annotations file ({ANNOTATION_FILE.name})...")

    results = mm.run_evaluation(
        perf_annotations=ANNOTATION_FILE,
        debug=True,
        save_dir=ROOT_DIR / "results",
        run_name="simple_example",
        level="note",
    )

    print(f"Evaluation Result: {json.dumps(results, indent=4)}")
    print(f"Detailed evaluation results saved in {ROOT_DIR / 'results'}")


if __name__ == "__main__":
    main()
