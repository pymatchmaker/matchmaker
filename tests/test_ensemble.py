import unittest
import warnings

import numpy as np

from matchmaker import EXAMPLE_PIECES, Matchmaker
from matchmaker.ensemble import EnsembleFollower, MergedStream
from matchmaker.matchmaker import AVAILABLE_METHODS, DEFAULT_KWARGS
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


if __name__ == "__main__":
    unittest.main()
