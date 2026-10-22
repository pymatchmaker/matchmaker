"""Tests for the switching Kalman filter audio follower."""

import unittest

import numpy as np
import partitura as pt

from matchmaker.prob.skf import (
    SwitchingKalmanFilterFollower,
    build_chord_sequence,
    build_spectral_templates,
)
from matchmaker.registry import REGISTRY


class TestSwitchingKalmanFilterFollower(unittest.TestCase):
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

    def test_chord_lengths_are_independent_of_time_signature(self):
        for beats, beat_type, expected_beats in (
            (4, 4, [0.0, 1.0, 2.0, 3.0]),
            (6, 8, [0.0, 2.0, 4.0, 6.0]),
            (2, 2, [0.0, 0.5, 1.0, 1.5]),
        ):
            with self.subTest(time_signature=(beats, beat_type)):
                part = pt.score.Part("P1", quarter_duration=480)
                part.add(pt.score.TimeSignature(beats, beat_type), start=0)
                for note_index in range(4):
                    part.add(
                        pt.score.Note(id=f"n{note_index}", step="C", octave=4),
                        start=note_index * 480,
                        end=(note_index + 1) * 480,
                    )
                _, lengths, onset_beats = build_chord_sequence(part.note_array())
                np.testing.assert_allclose(lengths, [0.25] * 4)
                np.testing.assert_allclose(onset_beats, expected_beats)

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
