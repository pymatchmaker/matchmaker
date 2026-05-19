#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the OnlineAlignment base class."""

import unittest

import numpy as np

from matchmaker.base import OnlineAlignment

RNG = np.random.RandomState(1984)


class _DummyTracker(OnlineAlignment):
    """Minimal subclass that advances current_index by one each step."""

    def step(self, features):
        self.current_index += 1


def _make_dummy(n_positions: int = 5, ref_dim: int = 4) -> _DummyTracker:
    ref = RNG.rand(n_positions, ref_dim).astype(np.float32)
    score_positions = np.arange(n_positions, dtype=np.float32) * 0.5
    return _DummyTracker(reference_features=ref, score_positions=score_positions)


class TestOnlineAlignmentContract(unittest.TestCase):
    def test_call_unimplemented_step_raises(self):
        """Base step() raises NotImplementedError; __call__ propagates it."""
        ref = RNG.rand(5, 4).astype(np.float32)
        score_positions = np.arange(5, dtype=np.float32)
        follower = OnlineAlignment(
            reference_features=ref, score_positions=score_positions
        )
        features, perf_time = RNG.rand(4).astype(np.float32), 0.0
        self.assertRaises(NotImplementedError, follower, features, perf_time)

    def test_call_returns_float_beat(self):
        """__call__ returns the current score beat as a Python float."""
        tr = _make_dummy()
        # current_position starts at 0 by default; step() advances by 1.
        # Subclass starts with current_position = 0, then __call__ does step()
        # → current_position = 1. score_positions[1] = 0.5.
        beat = tr(np.zeros(4, dtype=np.float32), 0.0)
        self.assertIsInstance(beat, float)
        self.assertAlmostEqual(beat, 0.5)

    def test_alignment_path_shape_and_units(self):
        """alignment_path is (2, T) with row 0 score beat, row 1 perf time (sec)."""
        tr = _make_dummy()
        for k in range(3):
            tr(np.zeros(4, dtype=np.float32), float(k) * 0.1)
        wp = tr.alignment_path
        self.assertEqual(wp.shape, (2, 3))
        self.assertEqual(wp.dtype, np.float64)
        # row 0: score beats from score_positions
        np.testing.assert_allclose(wp[0], [0.5, 1.0, 1.5])
        # row 1: perf times we passed in
        np.testing.assert_allclose(wp[1], [0.0, 0.1, 0.2], atol=1e-6)

    def test_alignment_path_empty(self):
        """alignment_path returns (2, 0) before any step."""
        tr = _make_dummy()
        wp = tr.alignment_path
        self.assertEqual(wp.shape, (2, 0))

    def test_is_still_following(self):
        """is_still_following returns True until last score position is reached."""
        tr = _make_dummy(n_positions=3)
        # current_position starts at 0; n_positions = 3 → still following until 2.
        self.assertTrue(tr.is_still_following())
        tr(np.zeros(4, dtype=np.float32), 0.0)  # cp = 1
        self.assertTrue(tr.is_still_following())
        tr(np.zeros(4, dtype=np.float32), 0.1)  # cp = 2 (= len - 1)
        self.assertFalse(tr.is_still_following())

    def test_run_yields_float_beats(self):
        """run() yields float beats from a queue and stops at STREAM_END."""
        from matchmaker.io.queue import RECVQueue
        from matchmaker.io.stream import STREAM_END

        ref = RNG.rand(5, 4).astype(np.float32)
        score_positions = np.arange(5, dtype=np.float32) * 0.5
        q = RECVQueue()
        for k in range(3):
            q.put((np.zeros(4, dtype=np.float32), float(k) * 0.1))
        q.put(STREAM_END)
        tr = _DummyTracker(
            reference_features=ref, score_positions=score_positions, queue=q
        )
        beats = list(tr.run())
        self.assertEqual(len(beats), 3)
        for b in beats:
            self.assertIsInstance(b, float)
        np.testing.assert_allclose(beats, [0.5, 1.0, 1.5])

    def test_current_perf_time_exposed(self):
        """The latest observation's perf time is stored on the instance."""
        tr = _make_dummy()
        tr(np.zeros(4, dtype=np.float32), 1.234)
        self.assertEqual(tr.current_perf_time, 1.234)


if __name__ == "__main__":
    unittest.main()
