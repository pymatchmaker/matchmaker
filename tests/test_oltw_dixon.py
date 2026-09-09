"""Tests for the frame- and event-level Dixon OLTW implementations."""

import unittest
from queue import Queue
from unittest.mock import patch

import numpy as np

from matchmaker.dp.oltw_dixon import (
    Direction,
    OnlineTimeWarpingDixonEvent,
    OnlineTimeWarpingDixonFrame,
)
from matchmaker.io.stream import STREAM_END


class TestOnlineTimeWarpingDixonFrame(unittest.TestCase):
    def test_initial_band_advances_both_pointers(self):
        """Both pointers advance until the initial search width is reached."""
        width = 5
        follower = OnlineTimeWarpingDixonFrame(
            reference_features=np.arange(10, dtype=np.float32).reshape(-1, 1),
            score_positions=np.arange(10, dtype=np.float32),
            ref_frame_to_beat=np.arange(10, dtype=np.float32),
            window_size=width,
            frame_rate=1,
        )

        for index in range(width):
            follower.step(np.array([float(index)], dtype=np.float32))
            self.assertEqual(follower.ref_pointer, index + 1)
            self.assertEqual(follower.input_pointer, index + 1)

    def test_initial_direction_does_not_force_alignment_point(self):
        width = 5
        follower = OnlineTimeWarpingDixonFrame(
            reference_features=np.arange(10, dtype=np.float32).reshape(-1, 1),
            score_positions=np.arange(10, dtype=np.float32),
            ref_frame_to_beat=np.arange(10, dtype=np.float32),
            window_size=width,
            frame_rate=1,
        )
        follower.ref_pointer = 2
        follower.input_pointer = 2
        follower.accumulated_costs = {
            (1, 1): 10.0,
            (0, 1): 1.0,
            (1, 0): 2.0,
        }

        self.assertEqual(follower.get_inc(), Direction.BOTH)
        self.assertEqual((follower.best_ref, follower.best_input), (0, 1))

    def test_alignment_path_has_one_position_per_performance_frame(self):
        follower = OnlineTimeWarpingDixonFrame(
            reference_features=np.arange(10, dtype=np.float32).reshape(-1, 1),
            score_positions=np.arange(10, dtype=np.float32),
            ref_frame_to_beat=np.arange(10, dtype=np.float32),
            window_size=5,
            frame_rate=1,
        )
        follower.input_pointer = 5
        follower.wp = np.array(
            [
                [0.0, 0.0, 2.0, 2.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            ]
        )

        np.testing.assert_array_equal(
            follower.alignment_path,
            np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [1.0, 1.0, 3.0, 3.0, 4.0],
                ]
            ),
        )

    def test_ref_direction_is_reevaluated_after_input_advance(self):
        follower = OnlineTimeWarpingDixonFrame(
            reference_features=np.arange(10, dtype=np.float32).reshape(-1, 1),
            score_positions=np.arange(10, dtype=np.float32),
            ref_frame_to_beat=np.arange(10, dtype=np.float32),
            window_size=5,
            frame_rate=1,
        )
        follower._initialized = True
        follower.ref_pointer = 1
        follower.input_pointer = 1

        with (
            patch.object(
                follower,
                "get_inc",
                side_effect=[Direction.BOTH, Direction.INPUT, Direction.INPUT],
            ) as get_inc,
            patch.object(follower, "advance_input") as advance_input,
            patch.object(follower, "advance_ref") as advance_ref,
            patch.object(follower, "_update_best_alignment"),
            patch.object(follower, "save_history"),
        ):
            follower.step(np.array([1.0], dtype=np.float32))

        self.assertEqual(get_inc.call_count, 3)
        advance_input.assert_called_once()
        advance_ref.assert_not_called()

    def test_run_yields_once_per_input_frame(self):
        rng = np.random.default_rng(0)
        inputs = rng.normal(size=(20, 2)).astype(np.float32)
        queue = Queue()
        for input_idx, features in enumerate(inputs):
            queue.put((features, input_idx / 2))
        queue.put(STREAM_END)

        follower = OnlineTimeWarpingDixonFrame(
            reference_features=rng.normal(size=(100, 2)).astype(np.float32),
            score_positions=np.arange(100, dtype=np.float32),
            ref_frame_to_beat=np.arange(100, dtype=np.float32),
            queue=queue,
            window_size=1.5,
            frame_rate=2,
        )

        positions = list(follower.run(verbose=False))

        self.assertEqual(len(positions), len(inputs))
        self.assertEqual(follower.input_pointer, len(inputs))


class TestOnlineTimeWarpingDixonEvent(unittest.TestCase):
    def make_follower(self, window_size=2, max_run_count=3):
        return OnlineTimeWarpingDixonEvent(
            reference_features=np.eye(8, dtype=np.float32),
            score_positions=np.arange(8, dtype=np.float32),
            window_size=window_size,
            max_run_count=max_run_count,
        )

    def test_corner_advances_both_dimensions(self):
        follower = self.make_follower(window_size=1)
        follower.step(np.eye(8, dtype=np.float32)[0])

        follower.step(np.eye(8, dtype=np.float32)[1])

        self.assertEqual((follower.j, follower.t), (1, 1))

    def test_max_run_count_forces_the_other_dimension(self):
        follower = self.make_follower(window_size=1, max_run_count=3)
        follower.j = 3
        follower.t = 3
        follower.run_count_ref = 3
        self.assertEqual(follower.get_inc(), Direction.INPUT)

        follower.run_count_ref = 0
        follower.run_count_input = 3
        self.assertEqual(follower.get_inc(), Direction.REF)

    def test_path_cost_normalizes_by_weighted_extent(self):
        follower = self.make_follower()
        follower._D[(2, 3)] = 12.0
        self.assertEqual(follower._norm_cost(2, 3), 2.0)
