"""alignment_path convention tests for every registered method.

The convention: ``alignment_path`` is a ``(2, T)`` array with
row 0 = performance time in seconds, row 1 = score position in beats.
Each tracker steps the example piece only until its ``alignment_path``
holds a couple of points (in ``setUpClass``); every test then checks a
single property of the resulting paths. The exact row units are pinned
down in ``test_base.py`` with controlled inputs.
"""

import unittest
import warnings

import numpy as np

from matchmaker import EXAMPLE_PIECES, Matchmaker
from matchmaker.matchmaker import AVAILABLE_METHODS

warnings.filterwarnings("ignore", module="partitura")


class TestAlignmentPathConvention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # given: every method has stepped the example piece just until its
        # alignment_path holds >= 2 points (enough to check the (2, T)
        # convention), or the stream ends, whichever comes first
        piece = EXAMPLE_PIECES["simple_mozart"]
        cases = [("midi", m, piece["midi"]) for m in AVAILABLE_METHODS["midi"]]
        cases += [("audio", m, piece["audio"]) for m in AVAILABLE_METHODS["audio"]]
        cls.paths = {}
        for input_type, method, perf_file in cases:
            mm = Matchmaker(
                score_file=piece["score"],
                performance_file=perf_file,
                input_type=input_type,
                method=method,
            )
            gen = mm.run(verbose=False)
            while np.asarray(mm.score_follower.alignment_path).shape[1] < 2:
                if next(gen, None) is None:
                    break
            gen.close()
            cls.paths[(input_type, method)] = np.asarray(
                mm.score_follower.alignment_path, dtype=float
            )

    def test_shape_is_two_rows(self):
        for key, wp in self.paths.items():
            with self.subTest(input_type=key[0], method=key[1]):
                # then: (2, T) with at least two points, all finite
                self.assertEqual(wp.shape[0], 2)
                self.assertGreater(wp.shape[1], 1)
                self.assertTrue(np.all(np.isfinite(wp)))

    def test_rows_are_non_negative(self):
        for key, wp in self.paths.items():
            with self.subTest(input_type=key[0], method=key[1]):
                # then: neither performance times nor score beats are negative
                self.assertTrue(np.all(wp >= 0))


if __name__ == "__main__":
    unittest.main()
