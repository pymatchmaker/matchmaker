"""Tests for the parangonar adapter (``matchmaker.external``).

The two OLTW matchers stand in for all four: they build in milliseconds,
while the transformer matchers load a torch checkpoint. What is tested here
is the adapter's side of the contract — state indices, confidence, and
correction — not parangonar's tracking itself.
"""

import unittest
import warnings

import numpy as np
import partitura

from matchmaker import EXAMPLE_PIECES
from matchmaker.external import OnlineParangonarAlignment, ParangonarProcessor

warnings.filterwarnings("ignore", module="partitura")


def _note(pitch, onset_sec, note_id="n0"):
    """One performance note in the shape ``ParangonarProcessor`` emits."""
    fields = [("onset_sec", "f4"), ("pitch", "i4"), ("id", "U256")]
    return np.array([(onset_sec, pitch, note_id)], dtype=fields)[0]


class TestOnlineParangonarAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        score = partitura.load_score(EXAMPLE_PIECES["simple_mozart"]["score"])
        cls.note_array = partitura.score.merge_parts(score.parts).note_array(
            include_grace_notes=True
        )

    def _follower(self, method="SL_OLTW", **kwargs):
        return OnlineParangonarAlignment(
            reference_features=self.note_array, method=method, **kwargs
        )

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            self._follower(method="nope")

    def test_states_are_the_matcher_states(self):
        """Matcher indices are used as-is, so the state grids must agree."""
        for method in ("SL_OLTW", "SLT_OLTW"):
            with self.subTest(method=method):
                follower = self._follower(method)
                np.testing.assert_allclose(
                    follower.score_positions,
                    np.asarray(follower.matcher.unique_onsets, dtype=np.float32),
                )

    def test_confidence_is_none_before_the_first_note(self):
        self.assertIsNone(self._follower().confidence())

    def test_confidence_is_a_probability_after_stepping(self):
        follower = self._follower()
        for i, pitch in enumerate(self.note_array["pitch"][:12]):
            follower.step(_note(int(pitch), 0.5 * i, f"n{i}"))
        confidence = follower.confidence()
        self.assertIsNotNone(confidence)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_set_position_moves_the_matcher_and_the_follower(self):
        for method in ("SL_OLTW", "SLT_OLTW"):
            with self.subTest(method=method):
                follower = self._follower(method)
                target = float(follower.score_positions[7])

                self.assertTrue(follower.set_position(target))
                self.assertEqual(follower.current_index, 7)
                self.assertEqual(follower.matcher.tracker.current_position, 7)
                self.assertEqual(follower.get_current_position(), target)

    def test_set_position_can_move_a_matcher_backwards(self):
        """The whole point: the OLTW playhead cannot retreat on its own."""
        follower = self._follower()
        for i, pitch in enumerate(self.note_array["pitch"][:20]):
            follower.step(_note(int(pitch), 0.5 * i, f"n{i}"))
        advanced = follower.current_index
        self.assertGreater(advanced, 1)

        follower.set_position(float(follower.score_positions[1]))
        self.assertEqual(follower.current_index, 1)

        follower.step(_note(int(self.note_array["pitch"][2]), 20.0, "back"))
        self.assertLess(follower.current_index, advanced)

    def test_set_position_declines_at_zero_strength(self):
        follower = self._follower()
        self.assertFalse(follower.set_position(4.0, strength=0.0))

    def test_step_clamps_to_the_last_state(self):
        follower = self._follower()
        follower.matcher = _RunawayMatcher(len(follower.score_positions) + 50)
        follower.step(_note(60, 0.0))

        self.assertEqual(follower.current_index, len(follower.score_positions) - 1)
        self.assertIsInstance(follower.get_current_position(), float)


class _RunawayMatcher:
    """Matcher stub returning a state index past the end of the score."""

    def __init__(self, index):
        self.index = index

    def __call__(self, performance_note):
        return self.index


class TestParangonarProcessor(unittest.TestCase):
    def test_chord_frame_is_refused(self):
        """It needs event-based framing; a chord in one frame is a mistake."""

        class _Msg:
            type = "note_on"
            velocity = 64

            def __init__(self, note):
                self.note = note

        processor = ParangonarProcessor()
        with self.assertRaises(ValueError):
            processor(([(_Msg(60), 0.0), (_Msg(64), 0.0)], 0.0))


if __name__ == "__main__":
    unittest.main()
