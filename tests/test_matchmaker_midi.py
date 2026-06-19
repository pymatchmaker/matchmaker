import json
import queue
import traceback
import unittest
import warnings

import numpy as np

from matchmaker import EXAMPLE_PIECES, Matchmaker
from matchmaker.features.midi import PitchProcessor
from matchmaker.io.midi import MidiStream
from matchmaker.matchmaker import AVAILABLE_METHODS
from matchmaker.prob.hmm import PitchHMM

warnings.filterwarnings("ignore", module="partitura")


class TestMatchmakerMidi(unittest.TestCase):
    def setUp(self):
        self.score_file = EXAMPLE_PIECES["bach_fugue"]["score"]
        self.perf_midi = EXAMPLE_PIECES["bach_fugue"]["midi"]
        self.match_file = EXAMPLE_PIECES["bach_fugue"]["match"]
        self.test_datasets = [
            {"name": "simple_mozart_k265_var1", **EXAMPLE_PIECES["simple_mozart"]},
            {"name": "bach_fugue_bwv_858", **EXAMPLE_PIECES["bach_fugue"]},
        ]

    def _run(self, mm):
        try:
            return list(mm.run(verbose=False))
        except queue.Empty as e:
            print(f"Error: {type(e)}, {e}")
            traceback.print_exc()
            mm._has_run = True
            return []

    def test_init_default(self):
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.perf_midi,
            input_type="midi",
        )
        self.assertIsInstance(mm.stream, MidiStream)
        self.assertIsInstance(mm.score_follower, PitchHMM)
        self.assertIsInstance(mm.processor, PitchProcessor)

    def test_run_yields_positions(self):
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.perf_midi,
            input_type="midi",
        )
        gen = mm.run(verbose=False)
        first_pos = next(gen)
        gen.close()
        self.assertIsInstance(first_pos, (int, float, np.integer))

    def test_all_methods_evaluation(self):
        for dataset in self.test_datasets:
            for method in ["hmm", "pthmm", "outerhmm", "arzt", "dixon"]:
                with self.subTest(dataset=dataset["name"], method=method):
                    mm = Matchmaker(
                        score_file=dataset["score"],
                        performance_file=dataset["midi"],
                        input_type="midi",
                        method=method,
                    )
                    self._run(mm)
                    results = mm.run_evaluation(gt=dataset["match"], debug=False)
                    current_test = f"{dataset['name']}_{method}_midi"
                    print(
                        f"[{current_test}] beat_0.5b={results['beat']['0.5b']:.3f}, ms_300ms={results['ms']['300ms']:.3f}"
                    )
                    for threshold in ["0.5b", "1b"]:
                        self.assertGreaterEqual(results["beat"][threshold], 0.3)
                    for threshold in ["300ms", "500ms", "1000ms"]:
                        self.assertGreaterEqual(results["ms"][threshold], 0.3)

    def test_alignment_path_valid(self):
        mm = Matchmaker(
            score_file=EXAMPLE_PIECES["simple_mozart"]["score"],
            performance_file=EXAMPLE_PIECES["simple_mozart"]["midi"],
            input_type="midi",
        )
        self._run(mm)
        wp = mm.score_follower.alignment_path
        # Shape must be (2, T) with T > 0
        self.assertEqual(wp.shape[0], 2)
        self.assertGreater(wp.shape[1], 0)
        # wp[0] = perf time (seconds): must be non-negative and non-decreasing
        perf_times = mm._wp_perf_to_seconds(wp[0].astype(float))
        self.assertTrue(np.all(perf_times >= 0))
        self.assertTrue(np.all(np.diff(perf_times) >= 0))
        # wp[1] = score beat: must be non-negative
        score_beats = wp[1].astype(float)
        self.assertTrue(np.all(score_beats >= 0))

    def test_evaluation_before_run_raises(self):
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.perf_midi,
            input_type="midi",
        )
        with self.assertRaises(ValueError):
            mm.run_evaluation(gt=self.match_file, debug=False)

    def test_evaluation_in_beat_domain(self):
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.perf_midi,
            input_type="midi",
        )
        self._run(mm)
        results = mm.run_evaluation(gt=self.match_file, domain="score", debug=False)
        print(f"RESULTS: {json.dumps(results, indent=4)}")
        self.assertIn("beat", results)
        self.assertIn("ms", results)
        self.assertGreaterEqual(results["beat"]["1b"], 0.3)

    def test_all_available_methods_smoke(self):
        piece = EXAMPLE_PIECES["simple_mozart"]
        for method in AVAILABLE_METHODS["midi"]:
            with self.subTest(method=method):
                mm = Matchmaker(
                    score_file=piece["score"],
                    performance_file=piece["midi"],
                    input_type="midi",
                    method=method,
                )
                gen = mm.run(verbose=False)
                first_pos = next(gen)
                gen.close()
                self.assertIsInstance(first_pos, (int, float, np.integer))


if __name__ == "__main__":
    unittest.main()
