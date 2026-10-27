import copy
import unittest

import numpy as np
import partitura as pt

from matchmaker.utils.eval import gt_from_match, resolve_gt

MATCH = "matchmaker/assets/Bach-fugue_bwv_858.match"


class TestGTFromStreamedPerformance(unittest.TestCase):
    """GT times come from the performance the follower streams when it holds the same notes."""

    @classmethod
    def setUpClass(cls):
        cls.perf, _, score = pt.load_match(MATCH, create_score=True)
        cls.score_notes = score.note_array()
        cls.secs, cls.beats = gt_from_match(MATCH, cls.score_notes)

    def shifted(self, offset):
        perf = copy.deepcopy(self.perf)
        for note in perf.performedparts[0].notes:
            note["note_on"] = float(note["note_on"]) + offset
        return perf

    def test_streamed_onsets_replace_the_match_times(self):
        secs, beats = resolve_gt(MATCH, self.score_notes, performance=self.shifted(4e-4))
        np.testing.assert_allclose(secs, self.secs + 4e-4)
        np.testing.assert_array_equal(beats, self.beats)

    def test_other_notes_fall_back_to_the_match_times(self):
        perf = self.shifted(4e-4)
        perf.performedparts[0].notes[0]["pitch"] += 1
        secs, beats = gt_from_match(MATCH, self.score_notes, performance=perf)
        np.testing.assert_array_equal(secs, self.secs)
        np.testing.assert_array_equal(beats, self.beats)


if __name__ == "__main__":
    unittest.main()
