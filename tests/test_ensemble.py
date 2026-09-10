import unittest
import warnings

import numpy as np

from matchmaker import EXAMPLE_PIECES, Matchmaker
from matchmaker.ensemble import EnsembleFollower, MergedStream
from matchmaker.matchmaker import (
    AVAILABLE_METHODS,
    DEFAULT_KWARGS,
    PARANGONAR_METHODS,
)
from matchmaker.registry import REGISTRY, STREAM_BUILDERS

warnings.filterwarnings("ignore", module="partitura")


class TestEnsembleRegistration(unittest.TestCase):
    """The ensemble is declared in methods.yaml like any other method."""

    def test_listed_for_both_input_types(self):
        for input_type in ("audio", "midi"):
            self.assertIn("ensemble", AVAILABLE_METHODS[input_type])
            self.assertIn("members", DEFAULT_KWARGS[input_type]["ensemble"])

    def test_spec_declares_the_merged_stream(self):
        for input_type in ("audio", "midi"):
            spec = REGISTRY.method(input_type, "ensemble")
            self.assertEqual(spec.cls_path, "matchmaker.ensemble:EnsembleFollower")
            self.assertEqual(spec.stream, "merged")
            self.assertIn(spec.stream, STREAM_BUILDERS)
            self.assertIn("members", spec.args)

    def test_anchor_holder_is_not_a_method(self):
        """``_ensemble`` only carries the YAML anchor."""
        self.assertNotIn("_ensemble", AVAILABLE_METHODS["audio"])

    def test_other_methods_build_the_standard_stream(self):
        self.assertIsNone(REGISTRY.method("midi", "pthmm").stream)


class TestEnsembleBuild(unittest.TestCase):
    def setUp(self):
        piece = EXAMPLE_PIECES["simple_mozart"]
        self.score_file = piece["score"]
        self.perf_midi = piece["midi"]
        self.perf_audio = piece["audio"]

    def _build(self, **kwargs):
        return Matchmaker(
            score_file=self.score_file,
            performance_file=self.perf_midi,
            input_type="midi",
            method="ensemble",
            **kwargs,
        )

    def test_builds_follower_stream_and_members(self):
        mm = self._build()
        self.assertIsInstance(mm.score_follower, EnsembleFollower)
        self.assertIsInstance(mm.stream, MergedStream)
        self.assertEqual(
            [member.name for member in mm.score_follower.members],
            ["pthmm", "outerhmm", "arzt"],
        )
        for member in mm.score_follower.members:
            self.assertEqual(member.modality, "midi")
            self.assertIsNotNone(member.processor)

    def test_members_are_shared_with_the_stream(self):
        """The stream and the follower must see the same member objects."""
        mm = self._build()
        self.assertIs(mm.score_follower.members, mm._ensemble_members_cache)

    def test_duplicate_member_names_are_disambiguated(self):
        mm = self._build(kwargs={"members": [{"method": "pthmm"}] * 2})
        self.assertEqual(
            [member.name for member in mm.score_follower.members],
            ["pthmm", "pthmm_2"],
        )

    def test_member_name_can_be_given(self):
        mm = self._build(
            kwargs={"members": [{"method": "pthmm", "name": "custom"}]},
        )
        self.assertEqual(mm.score_follower.members[0].name, "custom")

    def test_empty_members_is_an_error(self):
        with self.assertRaises(ValueError):
            self._build(kwargs={"members": []})

    def test_invalid_modality_is_an_error(self):
        with self.assertRaises(ValueError):
            self._build(
                kwargs={"members": [{"method": "pthmm", "input_type": "video"}]},
            )

    def test_mixed_modality_captures_both(self):
        """An audio member under a midi ensemble opens both raw streams."""
        mm = self._build(
            kwargs={
                "members": [
                    {"method": "pthmm"},
                    {"method": "arzt", "input_type": "audio"},
                ],
                "audio_performance_file": self.perf_audio,
            },
        )
        self.assertEqual(
            sorted(modality for modality, _ in mm.stream.children),
            ["audio", "midi"],
        )
        # frame_rate follows the modalities actually present, not input_type.
        self.assertGreater(mm.frame_rate, 1)

    def test_policy_is_configurable(self):
        # ``kwargs`` replaces the spec defaults wholesale (as for every method),
        # so an override must restate the members it still wants.
        mm = self._build(
            kwargs={
                "members": [{"method": "pthmm"}],
                "policy": "confidence_median",
            },
        )
        self.assertEqual(
            type(mm.score_follower.policy).__name__,
            "ConfidenceWeightedMedianPolicy",
        )

    def test_kwargs_replace_defaults(self):
        """Documents the branch-wide kwargs semantics for the ensemble."""
        with self.assertRaises(ValueError):
            self._build(kwargs={"policy": "agreement"})

    def test_alignment_path_rows_follow_the_convention(self):
        """Row 0 is performance seconds, row 1 score beats — as everywhere else."""
        mm = self._build(kwargs={"members": [{"method": "pthmm"}]})
        list(mm.run(verbose=False))
        wp = mm.score_follower.alignment_path

        self.assertEqual(wp.shape[0], 2)
        # The performance runs longer in seconds than the excerpt does in
        # beats, and only the score axis is quantized to score positions.
        score_positions = set(np.unique(mm.score_part.note_array()["onset_beat"]))
        self.assertTrue(set(wp[1]).issubset(score_positions))
        self.assertTrue(all(b >= a for a, b in zip(wp[0], wp[0][1:])))

    def test_run_yields_monotonic_positions(self):
        mm = self._build(kwargs={"members": [{"method": "pthmm"}]})
        positions = list(mm.run(verbose=False))

        self.assertGreater(len(positions), 0)
        for position in positions:
            self.assertIsInstance(position, float)
        self.assertTrue(
            all(b >= a for a, b in zip(positions, positions[1:])),
            "ensemble positions should not go backwards",
        )
        self.assertIsInstance(mm.score_follower.alignment_path, np.ndarray)


class TestEnsembleMidiFraming(unittest.TestCase):
    """The merged MIDI stream is framed for the hungriest member."""

    def setUp(self):
        piece = EXAMPLE_PIECES["simple_mozart"]
        self.score_file = piece["score"]
        self.perf_midi = piece["midi"]

    def _stream_polling_period(self, members, **kwargs):
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.perf_midi,
            input_type="midi",
            method="ensemble",
            kwargs={"members": members, **kwargs},
        )
        return dict(mm.stream.children)["midi"].polling_period

    def test_finest_member_period_wins(self):
        # pthmm asks for 0.01, arzt for 0.001
        self.assertEqual(
            self._stream_polling_period([{"method": "pthmm"}, {"method": "arzt"}]),
            0.001,
        )

    def test_event_based_member_makes_the_stream_event_based(self):
        """A parangonar member is fed one note per frame, so all members are."""
        self.assertIsNone(
            self._stream_polling_period([{"method": "pthmm"}, {"method": "SL_OLTW"}])
        )

    def test_override_that_would_starve_a_member_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self._stream_polling_period(
                [{"method": "pthmm"}, {"method": "SL_OLTW"}],
                midi_polling_period=0.01,
            )
        self.assertIn("SL_OLTW", str(ctx.exception))

    def test_period_can_be_overridden(self):
        self.assertEqual(
            self._stream_polling_period(
                [{"method": "pthmm"}], midi_polling_period=0.05
            ),
            0.05,
        )
        self.assertIsNone(
            self._stream_polling_period(
                [{"method": "pthmm"}], midi_polling_period=None
            )
        )


class TestParangonarMembers(unittest.TestCase):
    """The parangonar trackers work as ensemble members like any other method."""

    def setUp(self):
        piece = EXAMPLE_PIECES["simple_mozart"]
        self.score_file = piece["score"]
        self.perf_midi = piece["midi"]

    def _build(self, members, **kwargs):
        return Matchmaker(
            score_file=self.score_file,
            performance_file=self.perf_midi,
            input_type="midi",
            method="ensemble",
            kwargs={"members": members, **kwargs},
        )

    def test_every_parangonar_method_can_be_a_member(self):
        mm = self._build([{"method": m} for m in PARANGONAR_METHODS])
        self.assertEqual(
            [member.name for member in mm.score_follower.members],
            list(PARANGONAR_METHODS),
        )

    def test_mixed_ensemble_runs(self):
        mm = self._build([{"method": "pthmm"}, {"method": "SL_OLTW"}])
        positions = list(mm.run(verbose=False))

        self.assertGreater(len(positions), 0)
        self.assertGreater(mm.score_follower.alignment_path.shape[1], 0)

    def test_repeated_method_takes_its_own_kwargs(self):
        """The same tracker twice, configured differently, is two members."""
        mm = self._build(
            [
                {"method": "SL_OLTW", "name": "narrow", "kwargs": {"window_size": 5}},
                {"method": "SL_OLTW", "name": "wide", "kwargs": {"window_size": 50}},
            ]
        )
        widths = [
            member.follower.matcher.tracker.window_size
            for member in mm.score_follower.members
        ]
        self.assertEqual(widths, [5, 50])


if __name__ == "__main__":
    unittest.main()
