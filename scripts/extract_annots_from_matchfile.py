#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Example of how to extract the performed time of the beats
from a match file.
"""

import argparse
from pathlib import Path

import numpy as np
import partitura as pt
from partitura.musicanalysis.performance_codec import get_time_maps_from_alignment


def save_annotations_to_txt(performed_times, output_file):
    """
    Save times to a tab-separated text file.

    Args:
        performed_times (np.ndarray): Array of times
        output_file (str): Path to output file
    """
    with open(output_file, "w") as f:
        for perf_time in performed_times:
            line = f"{perf_time:.6f}\t{perf_time:.6f}\n"
            f.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, default="note", choices=["beat", "note"])
    args = parser.parse_args()
    level = args.level

    root_dir = Path(__file__).parent.parent
    # match_fn = root_dir / "matchmaker" / "assets" / "simple_mozart_k265_var1.match"
    match_fn = root_dir / "tests" / "resources" / "Chopin_op38_p01.match"

    # Load performance, score and alignment info
    # from match file. The performed MIDI files are aligned with
    # the audio, so the alignment also corresponds
    # to the alignment in the audio.
    perf, alignment, score = pt.load_match(
        filename=str(match_fn),
        create_score=True,
    )

    # Get a structured numpy array of notes in the
    # score and notes in the performance
    pnote_array = perf.note_array()
    snote_array = score.note_array()

    # Get interp1d object that map time in the performance
    # in seconds to time in the score in beats and the
    # other way around
    ptime_to_stime_map, stime_to_ptime_map = get_time_maps_from_alignment(
        ppart_or_note_array=pnote_array,
        spart_or_note_array=snote_array,
        alignment=alignment,
    )

    # To get, e.g., the times of the beats in the
    # performance
    if level == "beat":
        start_beat = np.ceil(snote_array["onset_beat"].min())
        end_beat = np.floor(snote_array["onset_beat"].max())
        beats = np.arange(start_beat, end_beat + 1)
        performed_beat_times = stime_to_ptime_map(beats)
        save_annotations_to_txt(
            performed_beat_times,
            match_fn.with_name(f"{match_fn.stem}_beat_annotations.txt"),
        )
    elif level == "note":
        unique_onsets = np.unique(snote_array["onset_beat"])
        performed_note_times = stime_to_ptime_map(unique_onsets)
        save_annotations_to_txt(
            performed_note_times,
            match_fn.with_name(f"{match_fn.stem}_note_annotations.txt"),
        )
