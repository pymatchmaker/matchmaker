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
