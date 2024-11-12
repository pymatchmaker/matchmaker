import unittest
import warnings

from matchmaker import Matchmaker
from matchmaker.dp import OnlineTimeWarpingArzt
from matchmaker.dp.oltw_dixon import OnlineTimeWarpingDixon
from matchmaker.features.audio import ChromagramProcessor
from matchmaker.features.midi import PitchIOIProcessor
from matchmaker.io.audio import AudioStream
from matchmaker.io.midi import MidiStream
from matchmaker.prob.hmm import PitchIOIHMM

warnings.filterwarnings("ignore", module="partitura")
warnings.filterwarnings("ignore", module="librosa")


class TestMatchmaker(unittest.TestCase):
    def setUp(self):
        # Set up paths to test files
        self.score_file = "./tests/resources/Bach-fugue_bwv_858.musicxml"
        self.performance_file_audio = "./tests/resources/Bach-fugue_bwv_858.mp3"
        self.performance_file_midi = "./tests/resources/Bach-fugue_bwv_858.mid"

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
        alignment_results = list(mm.run(verbose=False))

        # Then: the yielded result should be a float values
        for position_in_beat in alignment_results:
            self.assertIsInstance(position_in_beat, float)

        # And: the alignment result should be a list
        self.assertIsInstance(alignment_results, list)

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
