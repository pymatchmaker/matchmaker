"""Tests for the Dannenberg 1985 DP score follower.

Test scenario based on Figure 2.7 from:

    Bloch, J. & Dannenberg, R.B. (1985). Real-Time Computer Accompaniment
    of Keyboard Performances. Proc. ICMC, 279-289.

Score (three chords at beats 0, 1, 2):
    A0 = {A3, G4, E4}  = {57, 67, 64}
    A1 = {G4}           = {67}
    A2 = {A3, B3, C4}   = {57, 59, 60}

Performance: the score notes played in order with ~30-50 ms spacing
within each chord, plus an extra D4 (62) inserted before the third
chord.
"""

import unittest

import numpy as np

from matchmaker.dp.dannenberg import DPDannenberg


def _make_follower(**overrides):
    """Build a DPDannenberg for the three-chord Figure 2.7 example."""
    score_cevts = [{57, 67, 64}, {67}, {57, 59, 60}]
    score_positions = np.array([0.0, 1.0, 2.0])
    defaults = dict(
        reference_features=score_cevts,
        score_positions=score_positions,
        tempo=120.0,
        window=50,
        c_m=0.5,
        c_e=0.0,
        epsilon=0.090,
        epsilon_fraction=0.25,
        smoothing=0.5,
    )
    defaults.update(overrides)
    return DPDannenberg(**defaults)


def _feed(follower, notes):
    """Feed (pitch, time) pairs through __call__."""
    for pitch, t in notes:
        follower(np.array([pitch]), t)


class TestDPDannenbergInit(unittest.TestCase):
    def test_state_dimensions(self):
        f = _make_follower()
        self.assertEqual(f.n_states, 3)
        self.assertEqual(len(f._values), 3)
        self.assertEqual(len(f._used), 3)
        self.assertEqual(len(f._prev_values), 3)
        self.assertEqual(len(f._prev_used), 3)

    def test_initial_position(self):
        f = _make_follower()
        self.assertEqual(f.current_index, 0)
        self.assertAlmostEqual(f.get_current_position(), 0.0)

    def test_initial_tempo(self):
        f = _make_follower(tempo=60.0)
        self.assertAlmostEqual(f._beats_per_second, 1.0)

    def test_is_still_following_at_start(self):
        f = _make_follower()
        self.assertTrue(f.is_still_following())


class TestDPDannenbergStep(unittest.TestCase):
    """DP update logic using the three-chord example."""

    def test_first_chord_all_notes_matched(self):
        """All three notes of chord A0 should be matched."""
        f = _make_follower()
        _feed(f, [
            (57, 0.00),   # A3
            (67, 0.03),   # G4
            (64, 0.05),   # E4
        ])
        self.assertEqual(f.current_index, 0)
        self.assertAlmostEqual(f._values[0], 3.0)
        self.assertEqual(f._used[0], {57, 67, 64})

    def test_advance_to_second_chord(self):
        """After chord A0 is done, a G4 at a later time advances to A1."""
        f = _make_follower()
        _feed(f, [
            (57, 0.00),
            (67, 0.03),
            (64, 0.05),
            (67, 0.50),   # G4 — matches A1
        ])
        self.assertEqual(f.current_index, 1)

    def test_extra_note_no_penalty_default_c_e(self):
        """With c_e=0 an extra D4 does not decrease the best value."""
        f = _make_follower(c_e=0.0)
        _feed(f, [
            (57, 0.00),
            (67, 0.03),
            (64, 0.05),
            (67, 0.50),
        ])
        val_before = f._best_value
        _feed(f, [(62, 0.80)])  # D4 extra
        self.assertGreaterEqual(f._best_value, val_before)

    def test_extra_note_with_c_e_penalty(self):
        """With c_e > 0 an extra note not in B subtracts from z_val."""
        f = _make_follower(c_e=1.0)
        _feed(f, [
            (57, 0.00),
            (67, 0.03),
            (64, 0.05),
            (67, 0.50),
        ])
        val_before = float(f._values[f.current_index])
        _feed(f, [(62, 0.80)])  # D4 extra — penalty applied
        val_after = float(f._values[f.current_index])
        self.assertLess(val_after, val_before + 1)

    def test_full_performance_reaches_last_chord(self):
        """Play all chords; the follower should reach A2."""
        f = _make_follower()
        _feed(f, [
            (57, 0.00),   # A0: A3
            (67, 0.03),   # A0: G4
            (64, 0.05),   # A0: E4
            (67, 0.50),   # A1: G4
            (62, 0.80),   # extra D4
            (57, 1.00),   # A2: A3
            (59, 1.03),   # A2: B3
            (60, 1.05),   # A2: C4
        ])
        self.assertEqual(f.current_index, 2)

    def test_is_still_following_false_at_end(self):
        """Once the last score event is matched, is_still_following is False."""
        f = _make_follower()
        _feed(f, [
            (57, 0.00),
            (67, 0.03),
            (64, 0.05),
            (67, 0.50),
            (57, 1.00),
            (59, 1.03),
            (60, 1.05),
        ])
        self.assertFalse(f.is_still_following())

    def test_missing_note_c_m_penalty(self):
        """c_m penalty reduces diagonal advance value for missing notes."""
        f_no = _make_follower(c_m=0.0)
        f_hi = _make_follower(c_m=1.0)
        notes = [
            (57, 0.00),
            (67, 0.03),
            (67, 0.50),
        ]
        _feed(f_no, notes)
        _feed(f_hi, notes)
        # With c_m=0 no penalty for the missing E4 in A0; with c_m=1
        # the advance to row 1 loses 1 point for the missing note.
        self.assertGreater(f_no._values[1], f_hi._values[1])


class TestDPDannenbergIsNewCevt(unittest.TestCase):
    """Epsilon and epsilon-fraction grouping (Sec. 5)."""

    def test_first_note_is_always_new(self):
        f = _make_follower()
        self.assertTrue(f._is_new_cevt(0.0))

    def test_note_within_epsilon_is_same_cevt(self):
        f = _make_follower(epsilon=0.090)
        f._prev_note_time = 0.0
        self.assertFalse(f._is_new_cevt(0.05))

    def test_note_beyond_epsilon_is_new_cevt(self):
        f = _make_follower(epsilon=0.090)
        f._prev_note_time = 0.0
        self.assertTrue(f._is_new_cevt(0.10))

    def test_epsilon_fraction_triggers_new_cevt(self):
        f = _make_follower(epsilon=1.0, epsilon_fraction=0.1)
        f._prev_note_time = 0.0
        f._beats_per_second = 2.0
        f._best_row = 0
        # beat_gap = 1.0, predicted_dt = 0.5, threshold = 0.05
        self.assertTrue(f._is_new_cevt(0.06))

    def test_epsilon_fraction_no_trigger_when_close(self):
        f = _make_follower(epsilon=1.0, epsilon_fraction=0.5)
        f._prev_note_time = 0.0
        f._beats_per_second = 2.0
        f._best_row = 0
        # beat_gap = 1.0, predicted_dt = 0.5, threshold = 0.25
        self.assertFalse(f._is_new_cevt(0.20))

    def test_epsilon_fraction_skipped_at_last_row(self):
        f = _make_follower(epsilon=1.0, epsilon_fraction=0.01)
        f._prev_note_time = 0.0
        f._best_row = 2  # last row
        # No next row to compute beat_gap, so only epsilon matters.
        self.assertFalse(f._is_new_cevt(0.5))


class TestDPDannenbergGetCurrentPosition(unittest.TestCase):
    def test_position_at_anchor(self):
        f = _make_follower()
        f._anchor_beat = 1.0
        f._anchor_time = 0.5
        f.current_perf_time = 0.5
        self.assertAlmostEqual(f.get_current_position(), 1.0)

    def test_position_interpolated(self):
        f = _make_follower(tempo=120.0)
        f._anchor_beat = 0.0
        f._anchor_time = 0.0
        f.current_perf_time = 0.25
        pos = f.get_current_position()
        # 120 BPM = 2 beats/s, 0.25s drift = 0.5 beats
        self.assertAlmostEqual(pos, 0.5)

    def test_position_capped_at_next_beat(self):
        f = _make_follower(tempo=120.0)
        f._anchor_beat = 0.0
        f._anchor_time = 0.0
        f.current_index = 0
        f.current_perf_time = 5.0  # way ahead
        pos = f.get_current_position()
        # Capped at score_positions[1] = 1.0
        self.assertAlmostEqual(pos, 1.0)

    def test_position_not_capped_at_last_state(self):
        f = _make_follower(tempo=120.0)
        f._anchor_beat = 2.0
        f._anchor_time = 1.0
        f.current_index = 2  # last state
        f.current_perf_time = 1.5
        pos = f.get_current_position()
        # No cap since current_index + 1 == n_states
        self.assertAlmostEqual(pos, 3.0)


class TestDPDannenbergTempoTracking(unittest.TestCase):
    def test_tempo_updates_on_advance(self):
        f = _make_follower(tempo=120.0, smoothing=0.0)
        _feed(f, [
            (57, 0.00),
            (67, 0.03),
            (64, 0.05),
        ])
        initial_bps = f._beats_per_second
        _feed(f, [(67, 1.00)])  # advance to beat 1.0 at time 1.0
        # With smoothing=0, the tempo is purely the new rate.
        # beat advance = 1.0, time elapsed ~1.0 → rate ~ 1.0
        self.assertNotAlmostEqual(f._beats_per_second, initial_bps)

    def test_smoothing_blends_old_and_new(self):
        f = _make_follower(tempo=120.0, smoothing=0.5)
        _feed(f, [
            (57, 0.00),
            (67, 0.03),
            (64, 0.05),
        ])
        _feed(f, [(67, 1.00)])
        # smoothing=0.5 blends: 0.5 * 2.0 + 0.5 * new_rate
        bps = f._beats_per_second
        self.assertGreater(bps, 0.5)
        self.assertLess(bps, 3.0)


if __name__ == "__main__":
    unittest.main()
