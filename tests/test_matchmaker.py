import json
import queue
import traceback
import unittest
import warnings
from pathlib import Path

from matchmaker import Matchmaker
from matchmaker.dp import OnlineTimeWarpingArzt
from matchmaker.dp.oltw_dixon import OnlineTimeWarpingDixon
from matchmaker.features.audio import ChromagramProcessor
from matchmaker.features.midi import PitchIOIProcessor
from matchmaker.io.audio import AudioStream
from matchmaker.io.midi import MidiStream
from matchmaker.prob.hmm import PitchIOIHMM
from matchmaker.utils.eval import save_score_following_result

warnings.filterwarnings("ignore", module="partitura")
warnings.filterwarnings("ignore", module="librosa")


class TestMatchmaker(unittest.TestCase):
    def setUp(self):
        # Set up paths to test files
        self.score_file = "./tests/resources/Bach-fugue_bwv_858.musicxml"
        self.performance_file_audio = "./tests/resources/Bach-fugue_bwv_858.mp3"
        self.performance_file_midi = "./tests/resources/Bach-fugue_bwv_858.mid"
        self.performance_file_annotations = (
            "./tests/resources/Bach-fugue_bwv_858_annotations.txt"
        )

        # self.score_file = "./tests/resources/Chopin_op38.musicxml"
        # self.performance_file_audio = "./tests/resources/Chopin_op38_p01.wav"
        # self.performance_file_annotations = "./tests/resources/Chopin_op38_p01.tsv"

        # self.score_file = "./tests/resources/kv279_2.musicxml"
        # self.performance_file_audio = "./tests/resources/kv279_2.wav"
        # self.performance_file_annotations = "./tests/resources/kv279_2.tsv"

        # self.score_file = "./matchmaker/assets/mozart_k265_var1.musicxml"
        # self.performance_file_audio = "./matchmaker/assets/mozart_k265_var1.mp3"
        # self.performance_file_midi = "./matchmaker/assets/mozart_k265_var1.mid"
        # self.performance_file_annotations = (
        #     "./matchmaker/assets/mozart_k265_var1_annotations.txt"
        # )

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
        for method in ["arzt", "dixon"]:
            with self.subTest(method=method):
                mm = Matchmaker(
                    score_file=self.score_file,
                    performance_file=self.performance_file_audio,
                    wait=False,
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

                results = mm.run_evaluation(self.performance_file_annotations)
                current_test = f"{self._testMethodName}_{method}"
                print(f"[{current_test}] RESULTS: {json.dumps(results, indent=4)}")

                save_dir = Path("./tests/results")
                save_dir.mkdir(parents=True, exist_ok=True)
                score_annots = mm.build_score_annotations()
                save_score_following_result(
                    mm.score_follower,
                    save_dir,
                    score_annots,
                    self.performance_file_annotations,
                    name=f"{Path(self.performance_file_audio).stem}",
                )

                # Then: the results should at least be 0.7
                for threshold in ["300ms", "500ms", "1000ms"]:
                    self.assertGreaterEqual(results[threshold], 0.7)

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
            mm.run_evaluation(self.performance_file_annotations)

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

    def test_matchmaker_with_frame_rate(self):
        # Given: a Matchmaker instance with audio input
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.performance_file_audio,
            wait=False,
            input_type="audio",
            frame_rate=100,
        )

        # Then: the frame rate should be 100
        self.assertEqual(mm.frame_rate, 100)
        self.assertEqual(mm.score_follower.frame_rate, 100)

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
        self.assertIsInstance(mm.score_follower, PitchIOIHMM)
        self.assertIsInstance(mm.processor, PitchIOIProcessor)

    def test_matchmaker_midi_run(self):
        # Given: a Matchmaker instance with midi input
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.performance_file_midi,
            input_type="midi",
        )

        # When & Then: running the alignment process,
        # the yielded result should be a float values
        for position_in_beat in mm.run():
            print(f"Position in beat: {position_in_beat}")
            self.assertIsInstance(position_in_beat, float)
            if position_in_beat >= 130:
                break


if __name__ == "__main__":
    unittest.main()
