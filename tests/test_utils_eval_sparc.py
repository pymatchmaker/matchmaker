"""Adaptive SPARC on the same onset-wise windows used for tracking."""

import unittest
from unittest.mock import patch

import numpy as np

from matchmaker.utils.eval import (
    compute_sparc,
    evaluate_alignment,
    onset_wise_windows,
    spectral_arc_length,
)


class TestOnsetWiseWindows(unittest.TestCase):
    def test_unique_onsets_and_last_window_cover_the_end(self):
        windows = onset_wise_windows(np.array([44.0, 0.0, 0.0, 10.0, 20.0, np.nan]))
        self.assertEqual(
            [(w["t0"], w["t1"]) for w in windows],
            [(0.0, 30.0), (10.0, 40.0), (20.0, 44.0)],
        )
        self.assertEqual([w["n_points"] for w in windows], [4, 2, 2])
        self.assertEqual([w["n_onsets"] for w in windows], [3, 2, 2])
        self.assertTrue(windows[-1]["reaches_final_onset"])

    def test_short_performance_is_one_window(self):
        windows = onset_wise_windows(np.array([2.0, 4.0, 11.0]))
        self.assertEqual(len(windows), 1)
        self.assertEqual((windows[0]["t0"], windows[0]["t1"]), (2.0, 11.0))
        self.assertTrue(windows[0]["short_piece_fallback"])

    def test_invalid_duration_and_empty_onsets(self):
        self.assertEqual(onset_wise_windows(np.array([])), [])
        for duration in (0, -1, np.nan, np.inf):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                onset_wise_windows(np.array([0.0, 1.0]), duration)


class TestSpectralArcLength(unittest.TestCase):
    def test_matches_reference_sparc_for_a_gaussian_speed_profile(self):
        # Published example from the author's SPARC/scripts/smoothness.py.
        time = np.arange(-1.0, 1.0, 0.01)
        value = spectral_arc_length(np.exp(-5 * time**2), fs=100)
        self.assertAlmostEqual(value, -1.41403, places=5)

    def test_stationary_speed_is_undefined(self):
        self.assertIsNone(spectral_arc_length(np.zeros(1500), fs=50))
        self.assertIsNone(spectral_arc_length(np.array([1.0]), fs=50))


class TestPathSparc(unittest.TestCase):
    def details(self, beats, times, onsets):
        return compute_sparc(beats, times, onset_times=onsets, return_details=True)

    def test_alignment_result_has_compact_beat_metrics(self):
        times = np.arange(4.0)
        beats = times * 1.25
        result = evaluate_alignment(beats, times, times, times)
        self.assertEqual(
            list(result),
            [
                "mean_b",
                "median_b",
                "std_b",
                "accuracy_b",
                "skewness",
                "kurtosis",
                "smoothness",
            ],
        )
        self.assertEqual(result["mean_b"], 0.375)
        self.assertEqual(result["accuracy_b"]["0.5b"], 0.75)
        self.assertEqual(result["smoothness"], round(compute_sparc(beats, times), 4))

    def test_short_constant_speed_is_evaluated(self):
        times = np.arange(0.0, 10.02, 0.02)
        result = self.details(times, times, np.array([0.0, 5.0, 10.0]))
        self.assertIsNotNone(result["sparc"])
        self.assertEqual(result["sparc_n_windows"], 1)
        self.assertEqual(result["sparc_n_valid_windows"], 1)
        self.assertEqual(result["sparc_valid_window_fraction"], 1.0)
        self.assertAlmostEqual(result["sparc"], spectral_arc_length(np.ones(500), 50))

    def test_frozen_path_counts_every_window(self):
        times = np.arange(0.0, 60.02, 0.02)
        result = self.details(np.full_like(times, 1.7), times, np.arange(61.0))
        self.assertIsNone(result["sparc"])
        self.assertEqual(result["sparc_n_windows"], 31)
        self.assertEqual(result["sparc_n_valid_windows"], 0)
        self.assertEqual(result["sparc_valid_window_fraction"], 0.0)
        self.assertTrue(
            all(w["reason"] == "stationary" for w in result["sparc_windows"])
        )

    def test_partially_stationary_path_keeps_valid_fraction(self):
        times = np.arange(0.0, 60.02, 0.02)
        result = self.details(np.minimum(times, 30.0), times, np.arange(61.0))
        self.assertEqual(result["sparc_n_windows"], 31)
        self.assertEqual(result["sparc_n_valid_windows"], 30)
        self.assertEqual(result["sparc_windows"][-1]["reason"], "stationary")
        self.assertAlmostEqual(result["sparc_valid_window_fraction"], 30 / 31)

    def test_mean_is_unweighted_over_defined_windows(self):
        times = np.arange(0.0, 50.02, 0.02)
        with patch(
            "matchmaker.utils.eval.spectral_arc_length", side_effect=[-1.0, None, -5.0]
        ):
            result = self.details(times, times, np.array([0.0, 10.0, 20.0, 50.0]))
        self.assertEqual(result["sparc"], -3.0)
        self.assertEqual(result["sparc_n_windows"], 3)
        self.assertEqual(result["sparc_n_valid_windows"], 2)

    def test_redundant_reports_do_not_change_sparc(self):
        onsets = np.array([0.0, 4.0, 11.0, 18.0, 44.0])
        jump_times = np.arange(45.0)
        expected = self.details(jump_times, jump_times, onsets)
        for frame_rate in (25, 50):
            with self.subTest(frame_rate=frame_rate):
                times = np.arange(44 * frame_rate + 1) / frame_rate
                result = self.details(np.floor(times), times, onsets)
                self.assertEqual(result, expected)

    def test_uncovered_windows_are_not_filled_by_extrapolation(self):
        times = np.arange(0.0, 30.02, 0.02)
        result = self.details(times, times, np.arange(61.0))
        self.assertEqual(result["sparc_n_windows"], 31)
        self.assertEqual(result["sparc_n_valid_windows"], 1)
        self.assertTrue(
            all(w["reason"] == "uncovered" for w in result["sparc_windows"][1:])
        )

    def test_duplicate_timestamps_keep_the_last_decision(self):
        times = np.array([0.0, 1.0, 1.0, 2.0])
        beats = np.array([0.0, 99.0, 1.0, 2.0])
        result = self.details(beats, times, np.array([0.0, 2.0]))
        expected = compute_sparc(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
        self.assertAlmostEqual(result["sparc"], expected)

    def test_empty_path_has_no_defined_score(self):
        result = self.details(np.array([]), np.array([]), np.array([]))
        self.assertIsNone(result["sparc"])
        self.assertEqual(result["sparc_n_windows"], 0)
        self.assertEqual(result["sparc_n_valid_windows"], 0)
        self.assertIsNone(result["sparc_valid_window_fraction"])
