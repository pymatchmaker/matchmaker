#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Align MIDI performance with MusicXML score using parangonar and save as .match file.

This script uses parangonar (https://github.com/sildater/parangonar) to align
a MIDI performance file with a MusicXML score file and saves the result as a .match file.

References:
- Partitura: https://partitura.readthedocs.io/
- Parangonar: https://github.com/sildater/parangonar
"""

import argparse
from pathlib import Path

import partitura as pt
from parangonar.match import DualDTWNoteMatcher


def run_alignment_performance_to_score(
    midi_path: Path,
    score_path: Path,
    output_match_path: Path,
):
    midi_path = str(midi_path)
    score_path = str(score_path)

    perf = pt.load_performance_midi(midi_path)
    score_part = pt.load_score_as_part(score_path)

    score_note_array = pt.utils.note_array_from_part(
        score_part, include_grace_notes=True
    )
    perf_note_array = perf.note_array()

    matcher = DualDTWNoteMatcher()
    alignment = matcher(score_note_array, perf_note_array, score_part=score_part)

    # Save match file using Partitura
    # save_match requires: alignment, performance_data, score_data, out, and optionally assume_unfolded
    # score_data should be the original Part object (not unfolded) for proper saving
    print(f"Saving match file: {output_match_path}")
    pt.save_match(
        alignment=alignment,
        performance_data=perf,
        score_data=score_part,
        out=output_match_path.as_posix(),
        assume_unfolded=True,
    )

    print("\nMatch file saved successfully!")
    print(f"  Performance notes: {len(perf_note_array)}")
    print(f"  Score notes: {len(score_note_array)}")
    print(f"  Aligned notes: {len(alignment)}")
    print(f"  Output file: {output_match_path}")

    return output_match_path


if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent
    perf_path = Path(root_dir / "matchmaker" / "assets" / "simple_mozart_k265_var1.mid")
    score_path = Path(
        root_dir / "matchmaker" / "assets" / "simple_mozart_k265_var1.musicxml"
    )
    match_path = Path(
        root_dir / "matchmaker" / "assets" / "simple_mozart_k265_var1.match"
    )

    run_alignment_performance_to_score(perf_path, score_path, match_path)
