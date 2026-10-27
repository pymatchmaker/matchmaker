"""Tests for the Korzeniowski particle filter score follower."""

import unittest
import warnings

import numpy as np

from matchmaker import EXAMPLE_PIECES, Matchmaker
from matchmaker.features.processor import (
    KorzeniowskiObservation,
    KorzeniowskiScoreModel,
)
from matchmaker.prob.particle_filter_korzeniowski import ParticleFilterKorzeniowski

warnings.filterwarnings("ignore", module="partitura")
warnings.filterwarnings("ignore", module="librosa")


def _make_score_model() -> KorzeniowskiScoreModel:
    beat_grid = np.array([0.0, 0.5, 1.0], dtype=float)
    onset_positions = np.array([0.0, 0.5, 1.0], dtype=float)
    nearest_onsets = np.array([0.0, 0.5, 1.0], dtype=float)
    rest_mask = np.array([False, True, False])
    templates = np.ones((3, 4), dtype=float)
    active_notes = [np.array([60]), np.array([], dtype=int), np.array([64])]

    return KorzeniowskiScoreModel(
        beat_grid=beat_grid,
        onset_positions=onset_positions,
        nearest_onsets=nearest_onsets,
        rest_mask=rest_mask,
        templates=templates,
        active_notes=active_notes,
    )


class TestParticleFilterKorzeniowski(unittest.TestCase):
    def test_invalid_observation_type(self):
        with self.assertRaises(ValueError):
            ParticleFilterKorzeniowski(
                score_model=_make_score_model(),
                observation_type="invalid",
            )

    def test_direct_call_updates_alignment_path(self):
        follower = ParticleFilterKorzeniowski(
            score_model=_make_score_model(),
            observation_type="midi",
            num_particles=5,
        )
        observation = KorzeniowskiObservation(
            active_notes=np.array([60], dtype=np.int16),
            onset_notes=np.array([60], dtype=np.int16),
            loudness=-10.0,
        )

        beat = follower(observation, 0.0)

        self.assertIsInstance(beat, float)
        self.assertEqual(follower.alignment_path.shape, (2, 1))
        np.testing.assert_allclose(follower.alignment_path[0], [0.0])
        np.testing.assert_allclose(follower.alignment_path[1], [beat])

    def test_matchmaker_routes_pfkorz(self):
        piece = EXAMPLE_PIECES["simple_mozart"]
        cases = [
            (
                "audio",
                "audio",
                {
                    "processor": "korzeniowski",
                    "n_fft": 4096,
                    "win_length": 2048,
                    "num_particles": 5,
                },
            ),
            (
                "midi",
                "midi",
                {
                    "processor": "korzeniowski",
                    "piano_range": True,
                    "num_particles": 5,
                },
            ),
        ]

        for input_type, perf_key, kwargs in cases:
            with self.subTest(input_type=input_type):
                mm = Matchmaker(
                    score_file=piece["score"],
                    performance_file=piece[perf_key],
                    input_type=input_type,
                    method="pfkorz",
                    kwargs=kwargs,
                )

                self.assertIsInstance(
                    mm.score_follower,
                    ParticleFilterKorzeniowski,
                )
                self.assertEqual(mm.score_follower.num_particles, 5)

                gen = mm.run(verbose=False)
                first_pos = next(gen)
                gen.close()

                self.assertIsInstance(first_pos, (float, np.floating))


if __name__ == "__main__":
    unittest.main()