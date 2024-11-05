import unittest

from matchmaker import Matchmaker
from matchmaker.dp import OnlineTimeWarpingArzt
from matchmaker.dp.oltw_dixon import OnlineTimeWarpingDixon
from matchmaker.io.audio import AudioStream
from matchmaker.io.midi import MidiStream
from matchmaker.prob.hmm import PitchIOIHMM


class TestMatchmaker(unittest.TestCase):
    def setUp(self):
        # Set up paths to test files
        self.score_file = "./tests/resources/Bach-fugue_bwv_858.musicxml"
        self.performance_file_audio = "./tests/resources/Bach-fugue_bwv_858.mp3"
        self.performance_file_midi = "./tests/resources/Bach-fugue_bwv_858.mid"

    def test_matchmaker_audio_init(self):
        # Given: a Matchmaker instance with audio input
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.performance_file_audio,
            wait=False,
            input_type="audio",
        )

        # Then: the Matchmaker instance should be correctly initialized
        self.assertIsInstance(mm.stream, AudioStream)
        self.assertIsInstance(mm.score_follower, OnlineTimeWarpingDixon)

    def test_matchmaker_audio_alignment(self):
        # Given: a Matchmaker instance with audio input
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.performance_file_audio,
            wait=False,
            input_type="audio",
        )

        # When & Then: running the alignment process, the yielded result should be a float values
        for position_in_beat in mm.run():
            self.assertIsInstance(position_in_beat, float)

    def test_matchmaker_audio_with_result(self):
        # Given: a Matchmaker instance with audio input
        mm = Matchmaker(
            score_file=self.score_file,
            performance_file=self.performance_file_audio,
            wait=False,
            input_type="audio",
        )

        # When: running the alignment process (get the result)
        alignment_results = list(mm.run())

        # Then: the yielded result should be a float values
        for position_in_beat in alignment_results:
            self.assertIsInstance(position_in_beat, float)

        # And: the alignment result should be a list
        self.assertIsInstance(alignment_results, list)

    def test_audio_dixon(self):
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

    def test_audio_arzt(self):
        # Given: a Matchmaker instance with audio input and Dixon method
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

    def test_invalid_input_type(self):
        # Test Matchmaker with invalid input type
        with self.assertRaises(ValueError):
            Matchmaker(
                score_file=self.score_file,
                performance_file=self.performance_file_audio,
                input_type="midi",
            )

    def test_invalid_method(self):
        # Test Matchmaker with invalid method
        with self.assertRaises(ValueError):
            Matchmaker(
                score_file=self.score_file,
                performance_file=self.performance_file_audio,
                input_type="audio",
                method="invalid",
            )


if __name__ == "__main__":
    unittest.main()
