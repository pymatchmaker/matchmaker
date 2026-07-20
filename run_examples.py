import argparse
import datetime
from _queue import Empty
from pathlib import Path

from matchmaker import EXAMPLE_PIECES, Matchmaker

ROOT_DIR = Path(__file__).parent
_piece = EXAMPLE_PIECES["simple_mozart"]  # simple_mozart, bach_fugue
SCORE_FILE = Path(_piece["score"])
PERFORMANCE_AUDIO_FILE = Path(_piece["audio"])
PERFORMANCE_MIDI_FILE = Path(_piece["midi"])


def select_performance_file(input_mode):
    performance_file = (
        PERFORMANCE_MIDI_FILE if input_mode == "midi" else PERFORMANCE_AUDIO_FILE
    )
    print(f"Performance file: {performance_file.name}, with input mode: {input_mode}")
    return performance_file


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
    print(
        "Done."
    )


if __name__ == "__main__":
    main()
