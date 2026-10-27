#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the Arzt (2010) simple tempo model.
"""

import unittest
from unittest.mock import patch

import numpy as np

from matchmaker.dp.oltw_arzt import OnlineTimeWarpingArztFrame
from matchmaker.registry import REGISTRY
from matchmaker.utils.tempo_models import ArztTempoModel

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

    def _tracker(self, tempo_model=ArztTempoModel, **kwargs):
        return OnlineTimeWarpingArztFrame(
            reference_features=self.X,
            score_positions=self.score_positions,
            ref_frame_to_beat=self.ref_frame_to_beat,
            frame_rate=self.frame_rate,
            tempo_model=tempo_model,
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

    def test_invalid_tempo_configuration(self):
        for value in (0.0, -1.0, np.nan, np.inf, -np.inf):
            for parameter in ("init_beat_period", "tempo_min", "tempo_max"):
                with self.subTest(parameter=parameter, value=value):
                    with self.assertRaises(ValueError):
                        ArztTempoModel(**{parameter: value})
        with self.assertRaises(ValueError):
            ArztTempoModel(tempo_min=1.2, tempo_max=0.8)

    def test_invalid_estimates_preserve_last_valid_tempo(self):
        tracker = self._tracker()
        model = tracker.tempo_model
        with patch.object(model, "relative_tempo", return_value=1.1):
            model.before_step(tracker)
        for value in (None, 0.0, -1.0, np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                with patch.object(model, "relative_tempo", return_value=value):
                    model.before_step(tracker)
                model.alter(tracker, value)
                self.assertEqual(model.current_tempo, 1.1)
                self.assertEqual(model.beat_period, 0.5 / 1.1)
                self.assertEqual(model.n_inserted + model.n_deleted, 0)
                np.testing.assert_array_equal(tracker.V, self.X)

    def test_nonfinite_onsets_do_not_poison_tempo_history(self):
        model = ArztTempoModel()
        model(0.0, 0.0)
        for value in (np.nan, np.inf, -np.inf):
            model(value, 1.0)
            model(1.0, value)
        model(0.5, 1.0)
        self.assertEqual(model._history_perf, [0.0, 0.5])
        self.assertEqual(model._history_score, [0.0, 1.0])
        self.assertEqual(model.beat_period, 0.5)

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

    def test_backward_forward_refinement_tracks_with_shared_directions(self):
        for speed in (0.8, 1.0, 1.5):
            with self.subTest(speed=speed):
                tracker = self._tracker(backward_forward=True, max_run_count=3)
                beats = self._perform(tracker, speed, 400)
                expected = speed * np.arange(400) / self.frame_rate
                self.assertTrue(np.all(np.isfinite(beats)))
                self.assertLess(np.abs(beats[200:] - expected[200:]).max(), 0.5)

    def test_without_tempo_model(self):
        tracker = self._tracker(tempo_model=None)
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
        from matchmaker.utils.tempo_models import ArztTempoModel, TempoModel

        self.assertTrue(issubclass(ArztTempoModel, TempoModel))

        spec = REGISTRY.method("audio", "arzt_tempo")
        self.assertEqual(spec.cls_path, "matchmaker.dp:OnlineTimeWarpingArztFrame")
        self.assertIs(spec.default_kwargs.get("tempo_model"), ArztTempoModel)
        self.assertEqual(spec.default_kwargs.get("processor"), "chroma")

    def test_arzt_tempo_model_standalone_onsets(self):
        from matchmaker.utils.tempo_models import (
            ArztTempoModel,
            ArztWidmer2010TempoModel,
            make_tempo_model,
        )

        tm = ArztTempoModel(init_beat_period=0.5, n_onsets=5)
        self.assertIs(ArztWidmer2010TempoModel, ArztTempoModel)
        self.assertIs(make_tempo_model(tm), tm)
        built = make_tempo_model(ArztTempoModel, init_beat_period=0.6)
        self.assertIsInstance(built, ArztTempoModel)
        self.assertEqual(built.init_beat_period, 0.6)

        # First onset
        bp, est = tm(performed_onset=0.0, score_onset=0.0)
        self.assertEqual(bp, 0.5)
        self.assertEqual(est, 0.0)

        # Performance running 20% slower: performed IOI = 0.6s per score beat
        for i in range(1, 10):
            perf_t = i * 0.6
            score_b = float(i)
            bp, est = tm(performed_onset=perf_t, score_onset=score_b)

        # Relative tempo should reflect slower tempo (t < 1.0, beat_period > 0.5)
        self.assertLess(tm.current_tempo, 1.0)
        self.assertGreater(tm.beat_period, 0.5)

        # Reset
        tm.reset()
        self.assertEqual(tm.counter, 0)
        self.assertEqual(tm.current_tempo, 1.0)


if __name__ == "__main__":
    unittest.main()
