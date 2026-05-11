import json
import queue
import traceback
import unittest
import warnings
from pathlib import Path

import numpy as np

from matchmaker import EXAMPLE_PIECES, Matchmaker
from matchmaker.dp import OnlineTimeWarpingArzt
from matchmaker.dp.oltw_dixon import OnlineTimeWarpingDixon
from matchmaker.features.audio import ChromagramProcessor
from matchmaker.features.midi import PitchProcessor
from matchmaker.io.audio import AudioStream
from matchmaker.io.midi import MidiStream
from matchmaker.prob.hmm import PitchHMM, PitchIOIHMM
from matchmaker.prob.outer_product_hmm import OuterProductHMM
from matchmaker.prob.outer_product_hmm_audio import AudioOuterProductHMM

warnings.filterwarnings("ignore", module="partitura")
warnings.filterwarnings("ignore", module="librosa")


class TestMatchmaker(unittest.TestCase):
    def setUp(self):
        # Set up paths to test files
        self.score_file = EXAMPLE_PIECES["bach_fugue"]["score"]
        self.performance_file_audio = EXAMPLE_PIECES["bach_fugue"]["audio"]
        self.performance_file_midi = EXAMPLE_PIECES["bach_fugue"]["midi"]
        self.performance_file_annotations = EXAMPLE_PIECES["bach_fugue"]["annotations"]
        self.performance_file_beat_annotations = EXAMPLE_PIECES["bach_fugue"][
            "beat_annotations"
        ]

        self.test_datasets = [
            {"name": "simple_mozart_k265_var1", **EXAMPLE_PIECES["simple_mozart"]},
            {"name": "bach_fugue_bwv_858", **EXAMPLE_PIECES["bach_fugue"]},
        ]

    def test_matchmaker_audio_init(self):
        # When: a Matchmaker instance with audio input
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.performance_file_audio,
            wait=False,
            input_type="audio",
        )

        # Then: the Matchmaker instance should be correctly initialized
        self.assertIsInstance(mm.stream, AudioStream)
        self.assertIsInstance(mm.score_follower, OnlineTimeWarpingArzt)
        self.assertIsInstance(mm.processor, ChromagramProcessor)

    def test_matchmaker_audio_run(self):
        # Given: a Matchmaker instance with audio input
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.performance_file_audio,
            wait=False,
            input_type="audio",
        )

        # When & Then: running the alignment process, the yielded result should be a float values
        for position_in_beat in mm.run(verbose=False):
            self.assertIsInstance(position_in_beat, float)
            break

    def test_matchmaker_audio_run_with_result(self):
        # Given: a Matchmaker instance with audio input
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.performance_file_audio,
            wait=False,
            input_type="audio",
            method="dixon",
        )

        # When: running the alignment process (get the returned result)
        alignment_results = list(mm.run())

        # Then: the yielded result should be a float values
        for position_in_beat in alignment_results:
            self.assertIsInstance(position_in_beat, float)

        # And: the alignment result should be a list
        self.assertIsInstance(alignment_results, list)

    def test_matchmaker_audio_run_with_evaluation(self):
        for dataset in self.test_datasets:
            for method in ["arzt", "dixon", "outerhmm"]:
                with self.subTest(dataset=dataset["name"], method=method):
                    mm = Matchmaker(
                        score_file=dataset["score"],
                        performance_file=dataset["audio"],
                        input_type="audio",
                        method=method,
                    )

                    # When: running the alignment process
                    try:
                        alignment_positions = list(mm.run())
                    except queue.Empty as e:
                        print(f"Error: {type(e)}, {e}")
                        traceback.print_exc()
                        mm._has_run = True

                    current_test = f"{dataset['name']}_{method}"
                    results = mm.run_evaluation(
                        dataset["annotations"],
                        debug=False,
                        # save_dir=Path("./tests/results"),
                        # run_name=current_test,
                    )
                    print(f"[{current_test}] RESULTS: {json.dumps(results, indent=4)}")

                    # Then: the results should at least be 0.5
                    for threshold in ["300ms", "500ms", "1000ms"]:
                        self.assertGreaterEqual(results["ms"][threshold], 0.5)

    def test_matchmaker_audio_run_with_evaluation_cqt(self):
        # Given: a Matchmaker instance with audio input
        mm = Matchmaker(
            score_file=EXAMPLE_PIECES["simple_mozart"]["score"],
            performance_file=EXAMPLE_PIECES["simple_mozart"]["audio"],
            wait=False,
            input_type="audio",
            processor="cqt",
            method="arzt",
        )
        try:
            alignment_positions = list(mm.run())
        except queue.Empty as e:
            print(f"Error: {type(e)}, {e}")
            traceback.print_exc()
            mm._has_run = True

        results = mm.run_evaluation(
            EXAMPLE_PIECES["simple_mozart"]["annotations"],
            debug=False,
        )
        print(f"RESULTS: {json.dumps(results, indent=4)}")

        # Then: the results should at least be 0.5
        for threshold in ["300ms", "500ms", "1000ms"]:
            self.assertGreaterEqual(results["ms"][threshold], 0.5)

    def test_matchmaker_audio_run_with_evaluation_in_beats(self):
        # Given: a Matchmaker instance with audio input
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.performance_file_audio,
            wait=False,
            input_type="audio",
        )
        try:
            alignment_positions = list(mm.run())
        except queue.Empty as e:
            print(f"Error: {type(e)}, {e}")
            traceback.print_exc()
            mm._has_run = True

        results = mm.run_evaluation(
            self.performance_file_annotations, domain="score", debug=False
        )
        print(f"RESULTS: {json.dumps(results, indent=4)}")

        # Then: the results should at least be 0.5
        for threshold in ["0.3b", "0.5b", "1b"]:
            self.assertGreaterEqual(results["beat"][threshold], 0.5)

    def test_matchmaker_audio_run_with_evaluation_before_run(self):
        # Given: a Matchmaker instance with audio input
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.performance_file_audio,
            wait=False,
            input_type="audio",
        )

        # When: calling run_evaluation before run()
        with self.assertRaises(ValueError):
            mm.run_evaluation(self.performance_file_annotations, debug=False)

    def test_matchmaker_audio_dixon_init(self):
        # Given: a Matchmaker instance with audio input and Dixon method
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.performance_file_audio,
            wait=False,
            input_type="audio",
            method="dixon",
        )

        # Then: the Matchmaker instance should be correctly initialized
        self.assertIsInstance(mm.stream, AudioStream)
        self.assertIsInstance(mm.score_follower, OnlineTimeWarpingDixon)

    def test_matchmaker_audio_arzt_init(self):
        # When: a Matchmaker instance with audio input and Dixon method
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.performance_file_audio,
            wait=False,
            input_type="audio",
            method="arzt",
        )

        # Then: the Matchmaker instance should be correctly initialized
        self.assertIsInstance(mm.stream, AudioStream)
        self.assertIsInstance(mm.score_follower, OnlineTimeWarpingArzt)

    def test_matchmaker_audio_outerhmm_init(self):
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.performance_file_audio,
            input_type="audio",
            method="outerhmm",
        )

        self.assertIsInstance(mm.stream, AudioStream)
        self.assertIsInstance(mm.score_follower, AudioOuterProductHMM)

    def test_matchmaker_audio_outerhmm_run(self):
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.performance_file_audio,
            input_type="audio",
            method="outerhmm",
        )

        for position_in_beat in mm.run(verbose=False):
            self.assertIsInstance(position_in_beat, float)
            break

    def test_matchmaker_audio_rtf(self):
        for method in ["arzt", "dixon", "outerhmm"]:
            with self.subTest(method=method):
                mm = Matchmaker(
                    score_file=EXAMPLE_PIECES["simple_mozart"]["score"],
                    performance_file=EXAMPLE_PIECES["simple_mozart"]["audio"],
                    input_type="audio",
                    method=method,
                )
                list(mm.run(verbose=False))

                results = mm.run_evaluation(
                    EXAMPLE_PIECES["simple_mozart"]["annotations"], debug=False
                )
                self.assertIn("rtf", results)
                self.assertGreater(results["rtf"], 0)
                self.assertLess(results["rtf"], 1.0)

    def test_matchmaker_with_frame_rate(self):
        # Given: a Matchmaker instance with audio input
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.performance_file_audio,
            wait=False,
            input_type="audio",
            kwargs={"frame_rate": 50},
        )

        # Then: the frame rate should be 50
        self.assertEqual(mm.frame_rate, 50)
        self.assertEqual(mm.score_follower.frame_rate, 50)

    def test_matchmaker_invalid_input_type(self):
        # Test Matchmaker with invalid input type
        with self.assertRaises(ValueError):
            Matchmaker(
                score_file=self.score_file,
                performance_file=self.performance_file_audio,
                input_type="midi",
            )

    def test_matchmaker_invalid_method(self):
        # Test Matchmaker with invalid method
        with self.assertRaises(ValueError):
            Matchmaker(
                score_file=self.score_file,
                performance_file=self.performance_file_audio,
                input_type="audio",
                method="invalid",
            )

    def test_matchmaker_audio_run_with_distance_func(self):
        # Given: a Matchmaker instance with audio input
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.performance_file_audio,
            wait=False,
            input_type="audio",
        )

        # When & Then: distance function should be manhattan (= L1)
        self.assertEqual(mm.score_follower.distance_func.__class__.__name__, "L1")

    def test_matchmaker_midi_init(self):
        # When: a Matchmaker instance with midi input
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.performance_file_midi,
            input_type="midi",
        )

        # Then: the Matchmaker instance should be correctly initialized
        self.assertIsInstance(mm.stream, MidiStream)
        self.assertIsInstance(mm.score_follower, PitchHMM)
        self.assertIsInstance(mm.processor, PitchProcessor)

    def test_matchmaker_midi_run(self):
        # Given: a Matchmaker instance with midi input
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.performance_file_midi,
            input_type="midi",
        )

        # When & Then: running the alignment process,
        # the yielded result should be numeric (int state index for MIDI)
        for position_in_beat in mm.run():
            self.assertIsInstance(position_in_beat, (int, float, np.integer))
            if position_in_beat >= 10:
                break

    def test_matchmaker_midi_run_with_evaluation(self):
        """Test all MIDI methods: run + evaluate on test datasets."""
        for dataset in self.test_datasets:
            for method in ["hmm", "pthmm", "outerhmm", "arzt", "dixon"]:
                with self.subTest(dataset=dataset["name"], method=method):
                    mm = Matchmaker(
                        score_file=dataset["score"],
                        performance_file=dataset["midi"],
                        input_type="midi",
                        method=method,
                    )

                    try:
                        positions = list(mm.run())
                    except queue.Empty as e:
                        print(f"Error: {type(e)}, {e}")
                        traceback.print_exc()
                        mm._has_run = True

                    # All methods should produce positions
                    self.assertGreater(len(positions), 0)

                    # WP should be valid
                    wp = mm.score_follower.alignment_path
                    self.assertEqual(wp.shape[0], 2)
                    self.assertGreater(wp.shape[1], 0)

                    # Evaluate all methods
                    results = mm.run_evaluation(
                        dataset["annotations"],
                        debug=False,
                    )
                    current_test = f"{dataset['name']}_{method}_midi"
                    print(
                        f"[{current_test}] beat_0.5b={results['beat']['0.5b']:.3f}, ms_300ms={results['ms']['300ms']:.3f}"
                    )

                    for threshold in ["0.5b", "1b"]:
                        self.assertGreaterEqual(results["beat"][threshold], 0.3)
                    for threshold in ["300ms", "500ms", "1000ms"]:
                        self.assertGreaterEqual(results["ms"][threshold], 0.3)


if __name__ == "__main__":
    unittest.main()
