#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the Arzt (2010) Simple Tempo Model and Raphael/Jiang (2020) SKF.
"""

import unittest

import numpy as np

from matchmaker.dp.oltw_arzt import (
    OnlineTimeWarpingArztFrame,
    OnlineTimeWarpingArztTempoFrame,
)
from matchmaker.dp.oltw_dixon import Direction
from matchmaker.prob.skf import (
    SwitchingKalmanFilterFollower,
    build_chord_sequence,
    build_spectral_templates,
)
from matchmaker.registry import REGISTRY
from tests.utils import generate_example_sequences

RNG = np.random.RandomState(1984)


class TestArztTempoModel(unittest.TestCase):
    """Tests for the Arzt, Widmer & Dixon (2008) tracker with the
    Arzt & Widmer (2010 SMC) Simple Tempo Model."""

    def setUp(self):
        self.frame_rate = 50
        # 20 seconds of score audio -> 1000 frames, one onset per second
        n_frames = 1000
        n_features = 12
        rng = np.random.RandomState(1984)
        # each second of score is one random chord, held: distinct onsets,
        # steady frames between them
        chords = rng.rand(n_frames // self.frame_rate, n_features).astype(np.float32)
        self.X = np.repeat(chords, self.frame_rate, axis=0)
        self.X += rng.rand(*self.X.shape).astype(np.float32) * 0.05
        self.ref_frame_to_beat = np.arange(n_frames, dtype=np.float32) / self.frame_rate
        self.score_positions = np.arange(n_frames // self.frame_rate, dtype=np.float32)

    def make(self, **kwargs):
        return OnlineTimeWarpingArztTempoFrame(
            reference_features=self.X,
            score_positions=self.score_positions,
            ref_frame_to_beat=self.ref_frame_to_beat,
            frame_rate=self.frame_rate,
            window_size=4,
            **kwargs,
        )

    def follow(self, tracker, performance):
        for i, obs in enumerate(performance):
            beat = tracker(obs, i / self.frame_rate)
            self.assertIsInstance(beat, float)
        return tracker

    def test_initialization(self):
        tracker = self.make()
        self.assertNotIsInstance(tracker, OnlineTimeWarpingArztFrame)
        self.assertTrue(tracker.use_tempo_model)
        self.assertEqual(tracker.max_run_count, 6)
        self.assertEqual(tracker.STEP_WEIGHTS[Direction.BOTH][1], 2.0)
        self.assertEqual(tracker.STEP_WEIGHTS[Direction.REF][1], 1.3)
        self.assertEqual(tracker.current_relative_tempo, 1.0)
        self.assertEqual(tracker._consecutive_alterations, 0)
        self.assertEqual(len(tracker._backpointers), 0)
        self.assertTrue(tracker.is_onset_frame[0])
        self.assertTrue(tracker.is_onset_frame[50])
        self.assertFalse(tracker.is_onset_frame[25])

    def test_run_steps(self):
        tracker = self.follow(self.make(), self.X[:300])
        self.assertGreater(len(tracker._backpointers), 0)
        path = tracker.alignment_path
        self.assertEqual(path.shape, (2, 300))
        # the backward path ends in the current position and only moves forward
        bp = np.array(tracker._get_backward_path(100))
        self.assertEqual(tuple(bp[-1]), (tracker.best_ref, tracker.best_input))
        self.assertTrue(np.all(np.diff(bp, axis=0) >= 0))

    def test_slow_performance_stretches_score(self):
        # performance at half the score tempo: every score frame played twice
        tracker = self.follow(self.make(), np.repeat(self.X[:500], 2, axis=0))
        # t is the tempo relative to the notated score, not to the stretched one
        self.assertAlmostEqual(tracker.current_relative_tempo, 0.5, delta=0.1)
        self.assertGreater(tracker.n_inserted, 100)
        self.assertGreater(tracker.N_ref, len(self.X))
        # the alterations happen ahead of the forward path, so the reported
        # beat stays on the score
        self.assertAlmostEqual(tracker.get_current_position(), 10.0, delta=1.0)

    def test_fast_performance_compresses_score(self):
        # performance at 1.5 times the score tempo: every third frame dropped
        keep = np.arange(len(self.X)) % 3 != 2
        tracker = self.follow(self.make(), self.X[keep])
        self.assertGreater(tracker.current_relative_tempo, 1.2)
        self.assertGreater(tracker.n_deleted, 50)
        self.assertLess(tracker.N_ref, len(self.X))
        # no onset is lost by merging frames
        self.assertEqual(int(tracker.is_onset_frame.sum()), len(self.score_positions))

    def test_tempo_model_off(self):
        perf = np.repeat(self.X[:300], 2, axis=0)
        tracker = self.follow(self.make(use_tempo_model=False), perf)
        self.assertEqual(tracker.n_inserted + tracker.n_deleted, 0)
        self.assertEqual(tracker.N_ref, len(self.X))
        self.assertEqual(tracker.current_relative_tempo, 1.0)

    def test_deterministic_with_seed(self):
        perf = np.repeat(self.X[:300], 2, axis=0)
        a = self.follow(self.make(random_seed=7), perf).alignment_path
        b = self.follow(self.make(random_seed=7), perf).alignment_path
        np.testing.assert_array_equal(a, b)

    def test_reset(self):
        tracker = self.follow(self.make(), np.repeat(self.X[:300], 2, axis=0))
        self.assertNotEqual(tracker.N_ref, len(self.X))
        tracker.reset()
        self.assertEqual(tracker.current_index, 0)
        self.assertEqual(tracker.current_relative_tempo, 1.0)
        self.assertEqual(len(tracker._backpointers), 0)
        self.assertEqual(tracker.N_ref, len(self.X))
        np.testing.assert_array_equal(tracker.reference_features, self.X)

    def test_registry_lookup(self):
        spec = REGISTRY.method("audio", "arzt_tempo")
        self.assertEqual(spec.cls_path, "matchmaker.dp:OnlineTimeWarpingArztTempoFrame")
        self.assertTrue(spec.default_kwargs.get("use_tempo_model"))
        self.assertEqual(spec.default_kwargs.get("processor"), "lse")
        self.assertEqual(spec.default_kwargs.get("frame_rate"), 50)


class TestRaphaelSwitchingStateSpace(unittest.TestCase):
    """Tests for Raphael & Jiang (2020 ISMIR) Switching State-Space Model."""

    def setUp(self):
        # Create a simple synthetic note_array
        self.note_array = np.zeros(
            4,
            dtype=[
                ("pitch", "i4"),
                ("onset_beat", "f4"),
                ("duration_beat", "f4"),
            ],
        )
        self.note_array["pitch"] = [60, 64, 67, 72]  # C major chord arpeggio
        self.note_array["onset_beat"] = [0.0, 1.0, 2.0, 3.0]
        self.note_array["duration_beat"] = [1.0, 1.0, 1.0, 1.0]

    def test_build_chord_sequence(self):
        chords, lengths, onset_beats = build_chord_sequence(self.note_array)
        self.assertEqual(len(chords), 4)
        self.assertEqual(chords[0], [60])
        self.assertEqual(chords[1], [64])
        np.testing.assert_array_equal(onset_beats, [0.0, 1.0, 2.0, 3.0])
        # 1 beat = 0.25 whole note
        np.testing.assert_allclose(lengths, [0.25, 0.25, 0.25, 0.25])

    def test_build_spectral_templates(self):
        chords, _, _ = build_chord_sequence(self.note_array)
        templates = build_spectral_templates(chords, n_fft=512, sample_rate=8000)
        self.assertEqual(templates.shape[0], 4)
        self.assertEqual(templates.shape[1], 257)
        # Each template row should sum to 1
        np.testing.assert_allclose(templates.sum(axis=1), np.ones(4), atol=1e-5)

    def test_initialization_and_step(self):
        follower = SwitchingKalmanFilterFollower(
            reference_features=self.note_array,
            tempo=120.0,
            sample_rate=8000,
            n_fft=512,
            hop_length=128,
            max_hypotheses=50,
        )
        self.assertEqual(follower.K, 4)
        self.assertEqual(follower.current_index, 0)
        # Initial tempo: 240 / 120 = 2.0 s/whole-note
        self.assertAlmostEqual(follower.init_tempo, 2.0)

        # Process 20 frames
        delta = 128 / 8000.0
        for i in range(20):
            dummy_spectrum = np.random.rand(257).astype(np.float32)
            pos = follower(dummy_spectrum, perf_time=i * delta)
            self.assertIsInstance(pos, float)
            self.assertGreaterEqual(pos, 0.0)
            self.assertLessEqual(pos, 4.0)

        # Verify hypotheses count is bounded by max_hypotheses
        self.assertLessEqual(len(follower.hypotheses), 50)
        path = follower.alignment_path
        self.assertEqual(path.shape, (2, 20))

    def test_kalman_tempo_adaptation(self):
        follower = SwitchingKalmanFilterFollower(
            reference_features=self.note_array,
            tempo=120.0,
            sample_rate=8000,
            n_fft=512,
            hop_length=128,
        )
        # Test transition Kalman update directly:
        # Expected duration = length * tempo = 0.25 * 2.0 = 0.5s.
        # Suppose performer played chord 0 in 0.25s (twice as fast, tempo should decrease s/wn):
        fast_age = int(0.25 / follower.delta)
        mu_new, var_new = follower._kalman_update(
            mu_t_prev=2.0, var_t_prev=0.04, k_prev=0, age_prev=fast_age
        )
        self.assertLess(mu_new, 2.0)

        # Suppose performer played chord 0 in 1.0s (twice as slow, tempo should increase s/wn):
        slow_age = int(1.0 / follower.delta)
        mu_new_slow, var_new_slow = follower._kalman_update(
            mu_t_prev=2.0, var_t_prev=0.04, k_prev=0, age_prev=slow_age
        )
        self.assertGreater(mu_new_slow, 2.0)

    def test_registry_lookup(self):
        spec = REGISTRY.method("audio", "skf")
        self.assertEqual(
            spec.cls_path, "matchmaker.prob.skf:SwitchingKalmanFilterFollower"
        )


if __name__ == "__main__":
    unittest.main()
