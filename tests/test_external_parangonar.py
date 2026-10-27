#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the parangonar adapter in matchmaker.external."""

import unittest

import numpy as np

from matchmaker.external import OnlineParangonarAlignment, _initial_beat_period

NOTE_FIELDS = [
    ("onset_beat", "f4"),
    ("duration_beat", "f4"),
    ("onset_quarter", "f4"),
    ("duration_quarter", "f4"),
    ("pitch", "i4"),
    ("is_grace", "?"),
    ("id", "U256"),
]


def _score(grace_first: bool) -> np.ndarray:
    """Four eighth notes in 6/8 (one beat each), optionally after a grace note."""
    notes = [
        (float(i), 1.0, i / 2, 0.5, 60 + i, False, f"n{i}") for i in range(4)
    ]
    if grace_first:
        notes.insert(0, (0.0, 0.0, 0.0, 0.0, 59, True, "g0"))
    return np.array(notes, dtype=NOTE_FIELDS)


class TestInitialBeatPeriod(unittest.TestCase):
    def test_reads_the_first_note(self):
        # 90 quarters a minute, two beats to the quarter.
        self.assertAlmostEqual(_initial_beat_period(_score(False)), 60 / 90 / 2)

    def test_skips_a_leading_grace_note(self):
        period = _initial_beat_period(_score(True))
        self.assertTrue(np.isfinite(period))
        self.assertAlmostEqual(period, 60 / 90 / 2)

    def test_tempo_follower_starts_with_a_finite_tempo(self):
        follower = OnlineParangonarAlignment(_score(True), method="SLT_OLTW")
        self.assertTrue(np.isfinite(follower.matcher.init_tempo))


class TestStepClampsTheIndex(unittest.TestCase):
    def test_index_past_the_last_onset_is_clamped(self):
        follower = OnlineParangonarAlignment(_score(False), method="SL_OLTW")
        follower.matcher = lambda note: len(follower.score_positions) + 3
        follower(None, 0.0)
        self.assertEqual(follower.current_index, len(follower.score_positions) - 1)
        self.assertEqual(follower.current_position, 3.0)


if __name__ == "__main__":
    unittest.main()
