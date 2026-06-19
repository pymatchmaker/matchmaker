import json
import queue
import traceback
import unittest
import warnings

from matchmaker import EXAMPLE_PIECES, Matchmaker
from matchmaker.dp import OnlineTimeWarpingArzt
from matchmaker.features.audio import ChromagramProcessor
from matchmaker.io.audio import AudioStream
from matchmaker.matchmaker import AVAILABLE_METHODS

warnings.filterwarnings("ignore", module="partitura")
warnings.filterwarnings("ignore", module="librosa")


class TestMatchmakerAudio(unittest.TestCase):
    def setUp(self):
        self.score_file = EXAMPLE_PIECES["bach_fugue"]["score"]
        self.perf_audio = EXAMPLE_PIECES["bach_fugue"]["audio"]
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
            performance_file=self.perf_audio,
            wait=False,
            input_type="audio",
        )
        self.assertIsInstance(mm.stream, AudioStream)
        self.assertIsInstance(mm.score_follower, OnlineTimeWarpingArzt)
        self.assertIsInstance(mm.processor, ChromagramProcessor)

    def test_run_yields_floats(self):
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.perf_audio,
            wait=False,
            input_type="audio",
        )
        gen = mm.run(verbose=False)
        first_pos = next(gen)
        gen.close()
        self.assertIsInstance(first_pos, float)

    def test_all_methods_evaluation(self):
        for dataset in self.test_datasets:
            for method in ["arzt", "dixon", "outerhmm"]:
                with self.subTest(dataset=dataset["name"], method=method):
                    mm = Matchmaker(
                        score_file=dataset["score"],
                        performance_file=dataset["audio"],
                        input_type="audio",
                        method=method,
                    )
                    self._run(mm)
                    results = mm.run_evaluation(gt=dataset["match"], debug=False)
                    current_test = f"{dataset['name']}_{method}"
                    print(f"[{current_test}] RESULTS: {json.dumps(results, indent=4)}")
                    for threshold in ["300ms", "500ms", "1000ms"]:
                        self.assertGreaterEqual(results["ms"][threshold], 0.5)

    def test_evaluation_in_beat_domain(self):
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.perf_audio,
            wait=False,
            input_type="audio",
        )
        self._run(mm)
        results = mm.run_evaluation(gt=self.match_file, domain="score", debug=False)
        print(f"RESULTS: {json.dumps(results, indent=4)}")
        for threshold in ["0.3b", "0.5b", "1b"]:
            self.assertGreaterEqual(results["beat"][threshold], 0.5)

    def test_evaluation_before_run_raises(self):
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.perf_audio,
            wait=False,
            input_type="audio",
        )
        with self.assertRaises(ValueError):
            mm.run_evaluation(gt=self.match_file, debug=False)

    def test_rtf(self):
        for method in ["arzt", "dixon", "outerhmm"]:
            with self.subTest(method=method):
                mm = Matchmaker(
                    score_file=EXAMPLE_PIECES["simple_mozart"]["score"],
                    performance_file=EXAMPLE_PIECES["simple_mozart"]["audio"],
                    input_type="audio",
                    method=method,
                )
                self._run(mm)
                results = mm.run_evaluation(
                    gt=EXAMPLE_PIECES["simple_mozart"]["match"], debug=False
                )
                self.assertIn("rtf", results)
                self.assertGreater(results["rtf"], 0)
                self.assertLess(results["rtf"], 1.0)

    def test_cqt_processor_evaluation(self):
        mm = Matchmaker(
            score_file=EXAMPLE_PIECES["simple_mozart"]["score"],
            performance_file=EXAMPLE_PIECES["simple_mozart"]["audio"],
            wait=False,
            input_type="audio",
            processor="cqt",
            method="arzt",
        )
        self._run(mm)
        results = mm.run_evaluation(
            gt=EXAMPLE_PIECES["simple_mozart"]["match"], debug=False
        )
        print(f"RESULTS: {json.dumps(results, indent=4)}")
        for threshold in ["300ms", "500ms", "1000ms"]:
            self.assertGreaterEqual(results["ms"][threshold], 0.5)

    def test_frame_rate_propagation(self):
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.perf_audio,
            wait=False,
            input_type="audio",
            kwargs={"frame_rate": 50},
        )
        self.assertEqual(mm.frame_rate, 50)
        self.assertEqual(mm.score_follower.frame_rate, 50)

    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            Matchmaker(
                score_file=self.score_file,
                performance_file=self.perf_audio,
                input_type="audio",
                method="invalid",
            )

    def test_all_available_methods_smoke(self):
        piece = EXAMPLE_PIECES["simple_mozart"]
        for method in AVAILABLE_METHODS["audio"]:
            with self.subTest(method=method):
                mm = Matchmaker(
                    score_file=piece["score"],
                    performance_file=piece["audio"],
                    input_type="audio",
                    method=method,
                )
                gen = mm.run(verbose=False)
                first_pos = next(gen)
                gen.close()
                self.assertIsInstance(first_pos, float)


if __name__ == "__main__":
    unittest.main()
