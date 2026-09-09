import unittest
import warnings

import numpy as np

from matchmaker import EXAMPLE_PIECES, Matchmaker, register_method
from matchmaker.base import OnlineAlignment
from matchmaker.matchmaker import (
    AVAILABLE_METHODS,
    CUSTOM_METHODS,
    DEFAULT_KWARGS,
    unregister_method,
)

warnings.filterwarnings("ignore", module="partitura")


class MarchForward(OnlineAlignment):
    """Minimal follower: one score state per observation."""

    def step(self, features):
        self.current_index = min(self.current_index + 1, len(self.score_positions) - 1)


def build_march_forward(mm):
    return MarchForward(
        reference_features=mm.reference_features,
        score_positions=mm.score_positions,
        queue=mm.stream.queue,
    )


class TestRegisterMethod(unittest.TestCase):
    name = "test-march-forward"

    def setUp(self):
        self.score_file = EXAMPLE_PIECES["bach_fugue"]["score"]
        self.perf_midi = EXAMPLE_PIECES["bach_fugue"]["midi"]

    def tearDown(self):
        unregister_method(self.name, "midi")

    def _register(self, **overrides):
        kwargs = dict(
            input_type="midi",
            build_follower=build_march_forward,
            default_kwargs={"processor": "pitch", "piano_range": True},
        )
        kwargs.update(overrides)
        register_method(self.name, **kwargs)

    def test_registration_populates_the_registry(self):
        self._register()
        self.assertIn(self.name, AVAILABLE_METHODS["midi"])
        self.assertIn(("midi", self.name), CUSTOM_METHODS)
        self.assertEqual(DEFAULT_KWARGS["midi"][self.name]["processor"], "pitch")

    def test_registered_method_runs_through_matchmaker(self):
        self._register()
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.perf_midi,
            input_type="midi",
            method=self.name,
        )
        list(mm.run(verbose=False))

        self.assertIsInstance(mm.score_follower, MarchForward)
        path = mm.score_follower.alignment_path
        self.assertEqual(path.shape[0], 2)
        self.assertGreater(path.shape[1], 0)
        # Reference features default to the score note array.
        self.assertIn("onset_beat", mm.reference_features.dtype.names)

    def test_build_reference_hook_is_used(self):
        sentinel = np.arange(7)
        self._register(build_reference=lambda mm: sentinel)
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.perf_midi,
            input_type="midi",
            method=self.name,
        )
        np.testing.assert_array_equal(mm.reference_features, sentinel)

    def test_build_processor_hook_is_used(self):
        from matchmaker.features.midi import PianoRollProcessor

        made = PianoRollProcessor(piano_range=True)
        self._register(build_processor=lambda mm: made)
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.perf_midi,
            input_type="midi",
            method=self.name,
        )
        self.assertIs(mm.processor, made)

    def test_cannot_shadow_a_builtin(self):
        with self.assertRaises(ValueError):
            register_method(
                "pthmm", input_type="midi", build_follower=build_march_forward
            )

    def test_duplicate_registration_needs_overwrite(self):
        self._register()
        with self.assertRaises(ValueError):
            self._register()
        self._register(overwrite=True)  # explicit replacement is allowed

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            register_method(
                self.name, input_type="video", build_follower=build_march_forward
            )
        with self.assertRaises(TypeError):
            register_method(self.name, input_type="midi", build_follower="nope")

    def test_unregister_is_clean(self):
        self._register()
        unregister_method(self.name, "midi")
        self.assertNotIn(self.name, AVAILABLE_METHODS["midi"])
        self.assertNotIn(("midi", self.name), CUSTOM_METHODS)
        self.assertNotIn(self.name, DEFAULT_KWARGS["midi"])

    def test_builtin_methods_are_unaffected(self):
        self._register()
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.perf_midi,
            input_type="midi",
            method="pthmm",
        )
        list(mm.run(verbose=False))
        self.assertGreater(mm.score_follower.alignment_path.shape[1], 0)


class TestPublicAccessors(unittest.TestCase):
    def test_score_positions_is_ascending_and_unique(self):
        mm = Matchmaker(
            score_file=EXAMPLE_PIECES["bach_fugue"]["score"],
            performance_file=EXAMPLE_PIECES["bach_fugue"]["midi"],
            input_type="midi",
            method="pthmm",
        )
        positions = mm.score_positions
        self.assertGreater(len(positions), 0)
        self.assertTrue(np.all(np.diff(positions) > 0))


if __name__ == "__main__":
    unittest.main()
