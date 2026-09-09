import unittest

import numpy as np

from matchmaker.utils.errors import MatchmakerInvalidParameterTypeError
from matchmaker.utils.tempo_models import (
    LinearTempoExpectationsModel,
    LinearTempoModel,
)


def generate_performance(beat_periods, score_onsets):
    """Performed onsets that follow a given beat period curve exactly."""
    return np.concatenate(
        [[0.0], np.cumsum(np.diff(score_onsets) * beat_periods[:-1])],
    )


class TestLinearTempoExpectationsModel(unittest.TestCase):
    def setUp(self):
        self.score_onsets = np.arange(0, 20, 0.5)
        self.beat_periods = 0.6 + 0.15 * np.sin(self.score_onsets / 3.0)
        self.perf_onsets = generate_performance(self.beat_periods, self.score_onsets)
        self.tempo_curve = np.column_stack([self.score_onsets, self.beat_periods])

    def test_dummy_expectations(self):
        """Without a tempo curve the expectations are constant."""
        tempo_model = LinearTempoExpectationsModel(init_beat_period=0.6)

        self.assertTrue(tempo_model.has_tempo_expectations)
        self.assertEqual(tempo_model.scale_factor, 1.0)
        self.assertEqual(tempo_model.tempo_expectations(0.0), 0.6)
        self.assertEqual(tempo_model.tempo_expectations(123.4), 0.6)

    def test_expectations_are_anchored_to_init_beat_period(self):
        """The expected tempo at the first score onset is the initial tempo."""
        for tempo_expectations_func in (
            self.tempo_curve,
            lambda score_onset: 1.2 + 0.1 * score_onset,
        ):
            tempo_model = LinearTempoExpectationsModel(
                init_beat_period=0.6,
                tempo_expectations_func=tempo_expectations_func,
            )
            self.assertAlmostEqual(tempo_model.tempo_expectations(0.0), 0.6)

    def test_expectations_can_be_set_after_initialization(self):
        """Binding the tempo curve later re-anchors the expectations."""
        tempo_model = LinearTempoExpectationsModel(init_beat_period=0.6)
        tempo_model.tempo_expectations_func = lambda score_onset: 1.2

        self.assertAlmostEqual(tempo_model.scale_factor, 0.5)
        self.assertAlmostEqual(tempo_model.tempo_expectations(0.0), 0.6)

    def test_first_score_onset(self):
        """The anchor defaults to the initial score onset and can be moved."""
        tempo_model = LinearTempoExpectationsModel(
            init_beat_period=0.6,
            init_score_onset=4.0,
            tempo_expectations_func=lambda score_onset: 0.1 * score_onset,
        )
        self.assertEqual(tempo_model.first_score_onset, 4.0)
        self.assertAlmostEqual(tempo_model.scale_factor, 1.5)

        tempo_model.first_score_onset = 2.0
        self.assertAlmostEqual(tempo_model.scale_factor, 3.0)

    def test_unusable_anchor(self):
        """A zero-valued anchor does not produce a degenerate scale factor."""
        tempo_model = LinearTempoExpectationsModel(
            init_beat_period=0.6,
            tempo_expectations_func=lambda score_onset: 0.0,
        )
        self.assertEqual(tempo_model.scale_factor, 1.0)

    def test_invalid_expectations(self):
        with self.assertRaises(MatchmakerInvalidParameterTypeError):
            LinearTempoExpectationsModel(tempo_expectations_func="not a function")

    def test_beat_period_within_bounds(self):
        tempo_model = LinearTempoExpectationsModel(
            init_beat_period=0.6,
            tempo_expectations_func=self.tempo_curve,
            min_beat_period=0.25,
            max_beat_period=3.0,
        )
        for perf_onset, score_onset in zip(self.perf_onsets, self.score_onsets):
            beat_period, _ = tempo_model(perf_onset, score_onset)
            self.assertGreaterEqual(beat_period, 0.25)
            self.assertLessEqual(beat_period, 3.0)

    def test_tracks_expected_tempo_curve(self):
        """The expectations should beat the plain linear model on a known curve."""
        tempo_model = LinearTempoExpectationsModel(
            init_beat_period=0.6,
            tempo_expectations_func=self.tempo_curve,
        )
        baseline = LinearTempoModel(init_beat_period=0.6)

        estimated = np.array(
            [
                tempo_model(perf_onset, score_onset)[0]
                for perf_onset, score_onset in zip(self.perf_onsets, self.score_onsets)
            ]
        )
        baseline_estimated = np.array(
            [
                baseline(perf_onset, score_onset)[0]
                for perf_onset, score_onset in zip(self.perf_onsets, self.score_onsets)
            ]
        )

        error = np.abs(estimated - self.beat_periods).mean()
        baseline_error = np.abs(baseline_estimated - self.beat_periods).mean()

        self.assertLess(error, baseline_error)
        self.assertLess(error, 0.01)


if __name__ == "__main__":
    unittest.main()
