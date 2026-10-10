#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for Arzt (2010) Simple Tempo Model and Raphael/Jiang (2020) SKF.
"""

import unittest

import numpy as np

from matchmaker.dp.oltw_arzt import (
    OnlineTimeWarpingArztFrame,
    OnlineTimeWarpingArztTempoFrame,
)
from matchmaker.prob.skf import (
    SwitchingKalmanFilterFollower,
    build_chord_sequence,
    build_spectral_templates,
)
from matchmaker.registry import REGISTRY
from tests.utils import generate_example_sequences

RNG = np.random.RandomState(1984)


class TestArztTempoModel(unittest.TestCase):
    """Tests for Arzt, Widmer & Dixon (2008) OLTW with the Arzt & Widmer (2010) tempo model."""

    def setUp(self):
        # Generate synthetic reference and performance sequences
        self.frame_rate = 50
        # 20 seconds of score audio -> 1000 frames
        n_frames = 1000
        n_features = 12
        self.X = RNG.randn(n_frames, n_features).astype(np.float32)
        # 1 beat per second; every 50 frames is 1 beat
        self.ref_frame_to_beat = np.arange(n_frames, dtype=np.float32) / self.frame_rate
        # Score positions at beats 0.0, 1.0, 2.0, ..., 19.0
        self.score_positions = np.arange(20, dtype=np.float32)

    def _tracker(self, **kwargs):
        return OnlineTimeWarpingArztTempoFrame(
            reference_features=self.X,
            score_positions=self.score_positions,
            ref_frame_to_beat=self.ref_frame_to_beat,
            frame_rate=self.frame_rate,
            **kwargs,
        )

    def _perform(self, tracker, speed, n_steps):
        """Feed the reference played at ``speed`` times the score tempo."""
        beats = []
        for i in range(n_steps):
            idx = min(int(i * speed), len(self.X) - 1)
            obs = self.X[idx] + RNG.randn(12).astype(np.float32) * 0.01
            beats.append(tracker(obs, i / self.frame_rate))
        return np.array(beats)

    def test_initialization(self):
        tracker = self._tracker(window_size=5)
        self.assertTrue(tracker.use_tempo_model)
        self.assertEqual(tracker.c, 250)
        self.assertEqual(tracker.current_relative_tempo, 1.0)
        self.assertEqual(tracker.N, len(self.X))
        self.assertTrue(tracker.is_onset[0])
        self.assertTrue(tracker.is_onset[50])
        self.assertFalse(tracker.is_onset[25])

    def test_run_steps(self):
        tracker = self._tracker()
        beats = self._perform(tracker, 1.0, 200)
        self.assertTrue(all(isinstance(b, float) for b in beats))
        # Played at the score tempo: the position follows the input closely
        expected = np.arange(200) / self.frame_rate
        self.assertLess(np.abs(beats[100:] - expected[100:]).max(), 0.2)
        path = tracker.alignment_path
        self.assertEqual(path.shape, (2, 200))

    def test_tempo_model_follows_faster_performance(self):
        tracker = self._tracker()
        beats = self._perform(tracker, 1.5, 400)
        self.assertGreater(tracker.current_relative_tempo, 1.2)
        self.assertGreater(tracker.n_deleted, 0)
        self.assertEqual(tracker.n_inserted, 0)
        self.assertLess(tracker.N, len(self.X))
        expected = 1.5 * np.arange(400) / self.frame_rate
        self.assertLess(np.abs(beats[200:] - expected[200:]).max(), 0.5)

    def test_without_tempo_model(self):
        tracker = self._tracker(use_tempo_model=False)
        self._perform(tracker, 1.5, 200)
        self.assertEqual(tracker.n_deleted + tracker.n_inserted, 0)
        self.assertEqual(tracker.N, len(self.X))
        self.assertEqual(tracker.current_relative_tempo, 1.0)

    def test_reset(self):
        tracker = self._tracker()
        self._perform(tracker, 1.5, 300)
        self.assertLess(tracker.N, len(self.X))

        tracker.reset()
        self.assertEqual(tracker.current_index, 0)
        self.assertEqual(tracker.current_relative_tempo, 1.0)
        self.assertEqual(tracker.N, len(self.X))
        self.assertEqual(tracker.alignment_path.shape, (2, 0))
        np.testing.assert_array_equal(tracker.V, self.X)

    def test_registry_lookup(self):
        spec = REGISTRY.method("audio", "arzt_tempo")
        self.assertEqual(spec.cls_path, "matchmaker.dp:OnlineTimeWarpingArztTempoFrame")
        self.assertTrue(spec.default_kwargs.get("use_tempo_model"))
        self.assertEqual(spec.default_kwargs.get("processor"), "onset2008")


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
