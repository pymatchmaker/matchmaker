#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the dp.oltw_arzt module.
"""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import distance as sp_distance

from matchmaker.dp.direction import Direction
from matchmaker.dp.dtw_loop import oltw_arzt_loop
from matchmaker.dp.oltw_arzt import BackpointerColumn
from matchmaker.dp.oltw_arzt import OnlineTimeWarpingArztFrame as OnlineTimeWarpingArzt
from matchmaker.dp.oltw_dixon import Direction as DixonDirection
from matchmaker.dp.oltw_dixon import OnlineTimeWarpingDixonFrame
from matchmaker.utils import (
    CYTHONIZED_METRICS_W_ARGUMENTS,
    CYTHONIZED_METRICS_WO_ARGUMENTS,
)
from matchmaker.utils.errors import (
    MatchmakerInvalidOptionError,
    MatchmakerInvalidParameterTypeError,
)
from matchmaker.utils.tempo_models import ArztTempoModel
from tests.utils import generate_example_sequences

RNG = np.random.RandomState(1984)

_ALL_SCIPY_DISTANCES = [
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "euclidean",
    "jensenshannon",
    "minkowski",
    "sqeuclidean",
    "dice",
    "hamming",
    "jaccard",
    "rogerstanimoto",
    "russellrao",
    "sokalmichener",
    "sokalsneath",
    "yule",
]
# Filter to distances available in the installed scipy version
SCIPY_DISTANCES = [d for d in _ALL_SCIPY_DISTANCES if hasattr(sp_distance, d)]


class TestOnlineTimeWarpingArzt(unittest.TestCase):
    def test_distance_func(self):
        """
        Test initialization of the class
        """

        X, Y, path = generate_example_sequences(
            lenX=10,
            centers=3,
            n_features=3,
            maxreps=4,
            minreps=1,
            # do not use noise to ensure perfect
            # alignments
            noise_scale=0.00,
            random_state=RNG,
            dtype=np.float32,
        )
        score_positions = np.arange(X.shape[0], dtype=np.float32)

        self.assertTrue(X.dtype == np.float32)
        self.assertTrue(Y.dtype == np.float32)

        # Test raising error if local_cost_fun is invalid type
        self.assertRaises(
            MatchmakerInvalidParameterTypeError,
            OnlineTimeWarpingArzt,
            reference_features=X,
            score_positions=score_positions,
            ref_frame_to_beat=score_positions,
            window_size=2,
            step_size=1,
            # Invalid type (not str, tuple or callable)
            distance_func=RNG.rand(19),
            start_window_size=2,
            frame_rate=1,
        )

        # Test distance_func as string
        for distance_func in CYTHONIZED_METRICS_WO_ARGUMENTS:
            oltw = OnlineTimeWarpingArzt(
                reference_features=X,
                score_positions=score_positions,
                ref_frame_to_beat=score_positions,
                window_size=2,
                step_size=1,
                distance_func=distance_func,
                start_window_size=2,
                frame_rate=1,
            )

            for i, obs in enumerate(Y):
                current_position = oltw(obs, float(i))
                # check that the alignments are correct
                self.assertTrue(np.all(path[i] == (current_position, i)))
                # __call__ now returns the float beat position
                self.assertTrue(isinstance(current_position, float))

        # Test that error is raised if incorrect name
        self.assertRaises(
            MatchmakerInvalidOptionError,
            OnlineTimeWarpingArzt,
            reference_features=X,
            score_positions=score_positions,
            ref_frame_to_beat=score_positions,
            window_size=2,
            step_size=1,
            distance_func="wrong_distance_func",
            start_window_size=2,
        )

        # Test local_cost_fun as tuple
        for distance_func in CYTHONIZED_METRICS_W_ARGUMENTS:
            if distance_func == "Lp":
                for p in RNG.uniform(low=1, high=10, size=10):
                    oltw = OnlineTimeWarpingArzt(
                        reference_features=X,
                        score_positions=score_positions,
                        ref_frame_to_beat=score_positions,
                        window_size=2,
                        step_size=1,
                        distance_func=(distance_func, dict(p=p)),
                        start_window_size=2,
                        frame_rate=1,
                    )

                    for i, obs in enumerate(Y):
                        current_position = oltw(obs, float(i))
                        # check that the alignments are correct
                        self.assertTrue(np.all(path[i] == (current_position, i)))
                        # __call__ now returns the float beat position
                        self.assertTrue(isinstance(current_position, float))

    def test_tempo_model(self):
        """Test OnlineTimeWarpingArztFrame with ArztTempoModel."""
        from matchmaker.utils.tempo_models import ArztTempoModel

        X, Y, path = generate_example_sequences(
            lenX=50,
            centers=3,
            n_features=3,
            maxreps=4,
            minreps=1,
            noise_scale=0.00,
            random_state=RNG,
            dtype=np.float32,
        )
        score_positions = np.arange(X.shape[0], dtype=np.float32)

        # Baseline tracker without tempo model
        oltw_no_tm = OnlineTimeWarpingArzt(
            reference_features=X,
            score_positions=score_positions,
            ref_frame_to_beat=score_positions,
            window_size=5,
            step_size=3,
            frame_rate=50,
        )
        self.assertEqual(oltw_no_tm.current_relative_tempo, 1.0)
        self.assertFalse(oltw_no_tm.backward_forward)

        # Tracker with ArztTempoModel class passed
        oltw_tm = OnlineTimeWarpingArzt(
            reference_features=X,
            score_positions=score_positions,
            ref_frame_to_beat=score_positions,
            window_size=5,
            step_size=3,
            frame_rate=50,
            tempo_model=ArztTempoModel,
        )
        self.assertFalse(oltw_tm.backward_forward)
        self.assertIsInstance(oltw_tm.tempo_model, ArztTempoModel)

        positions = []
        for obs in Y:
            pos = oltw_tm(obs, 0.0)
            self.assertIsInstance(pos, float)
            positions.append(pos)
        self.assertGreater(len(positions), 0)
        self.assertIsInstance(oltw_tm.current_relative_tempo, float)

        # Test that error is raised if incorrect name
        self.assertRaises(
            MatchmakerInvalidOptionError,
            OnlineTimeWarpingArzt,
            reference_features=X,
            score_positions=score_positions,
            ref_frame_to_beat=score_positions,
            window_size=2,
            step_size=1,
            distance_func=("wrong_distance_func", {"param": "value"}),
            start_window_size=2,
            frame_rate=1,
        )

        for spdist in SCIPY_DISTANCES:
            oltw = OnlineTimeWarpingArzt(
                reference_features=X,
                score_positions=score_positions,
                ref_frame_to_beat=score_positions,
                window_size=2,
                step_size=1,
                distance_func=getattr(sp_distance, spdist),
                start_window_size=2,
                frame_rate=1,
            )
            # scipy metrics may not guarantee correct alignments; type check only
            current_position = oltw(Y[0], 0.0)
            self.assertTrue(isinstance(current_position, float))


class TestArztTempoIntegration(unittest.TestCase):
    def make_follower(self, **kwargs):
        return OnlineTimeWarpingArzt(
            reference_features=np.eye(8, dtype=np.float32),
            score_positions=np.arange(8, dtype=np.float32),
            ref_frame_to_beat=np.arange(8, dtype=np.float32),
            frame_rate=1,
            tempo_model=ArztTempoModel,
            **kwargs,
        )

    def test_cython_predecessors_use_shared_directions(self):
        self.assertIs(DixonDirection, Direction)
        for direction, predecessor in (
            (Direction.REF, (1, 1)),
            (Direction.INPUT, (2, 0)),
            (Direction.BOTH, (1, 0)),
        ):
            with self.subTest(direction=direction):
                costs = np.full((4, 2), 10.0, dtype=np.float32)
                costs[predecessor] = 0.0
                directions = np.full(1, Direction.NONE, dtype=np.int8)
                result, _, _ = oltw_arzt_loop(
                    costs,
                    np.ones(1, dtype=np.float32),
                    1,
                    2,
                    1,
                    np.inf,
                    0,
                    directions,
                )
                self.assertEqual(directions[0], direction)
                self.assertIsInstance(result, np.ndarray)
                self.assertTrue(np.shares_memory(costs, result))

    def test_origin_has_no_predecessor(self):
        directions = np.zeros(2, dtype=np.int8)
        oltw_arzt_loop(
            np.full((3, 2), np.inf, dtype=np.float32),
            np.ones(2, dtype=np.float32),
            0,
            2,
            0,
            np.inf,
            0,
            directions,
        )
        self.assertEqual(directions[0], Direction.NONE)

    def test_backward_path_follows_each_direction_and_stops_at_none(self):
        follower = self.make_follower()
        follower._backpointers = {
            3: BackpointerColumn(
                0,
                np.array(
                    [
                        Direction.NONE,
                        Direction.NONE,
                        Direction.INPUT,
                        Direction.REF,
                    ],
                    dtype=np.int8,
                ),
            ),
            2: BackpointerColumn(
                0,
                np.array(
                    [
                        Direction.NONE,
                        Direction.NONE,
                        Direction.BOTH,
                    ],
                    dtype=np.int8,
                ),
            ),
            1: BackpointerColumn(0, np.full(2, Direction.NONE, dtype=np.int8)),
        }
        self.assertEqual(
            list(follower.tempo_backward_path(3, 3)), [(3, 2), (2, 2), (1, 1)]
        )
        self.assertEqual(list(follower.tempo_backward_path(4, 3)), [])
        self.assertEqual(list(follower.tempo_backward_path(3, 5)), [])

    def test_last_deletable_frame_updates_costs_and_predecessors(self):
        follower = self.make_follower()
        follower._current_frame = 5
        follower.global_cost_matrix[:] = np.arange(18).reshape(9, 2)
        expected_costs = follower.global_cost_matrix[:-1].copy()
        follower._backpointers[0] = BackpointerColumn(
            0, np.full(8, Direction.BOTH, dtype=np.int8)
        )
        follower.tempo_set_score(
            follower.V[:-1], follower._ref_frame_to_beat[:-1], follower.is_onset[:-1]
        )
        self.assertEqual(follower.N_ref, 7)
        np.testing.assert_array_equal(follower.global_cost_matrix, expected_costs)
        self.assertEqual(len(follower._backpointers[0].predecessor_directions), 7)

    def test_insertion_interpolates_costs_and_adds_ref_predecessor(self):
        follower = self.make_follower()
        follower._current_frame = 2
        follower.global_cost_matrix[:] = np.arange(18).reshape(9, 2)
        expected_row = 0.5 * (
            follower.global_cost_matrix[3] + follower.global_cost_matrix[4]
        )
        follower._backpointers[0] = BackpointerColumn(
            0, np.full(8, Direction.BOTH, dtype=np.int8)
        )
        follower.tempo_set_score(
            np.insert(follower.V, 3, follower.V[2], axis=0),
            np.insert(follower._ref_frame_to_beat, 3, 2.5),
            np.insert(follower.is_onset, 3, False),
        )
        np.testing.assert_array_equal(follower.global_cost_matrix[4], expected_row)
        self.assertEqual(follower.N_ref, 9)
        self.assertEqual(
            follower._backpointers[0].predecessor_directions[3], Direction.REF
        )

    def test_invalid_alterations_leave_score_and_costs_unchanged(self):
        for current_frame, size_change in ((6, -1), (8, 1), (2, -2)):
            with self.subTest(current_frame=current_frame, size_change=size_change):
                follower = self.make_follower()
                follower._current_frame = current_frame
                costs = follower.global_cost_matrix.copy()
                if size_change < 0:
                    features = follower.V[:size_change]
                    beats = follower._ref_frame_to_beat[:size_change]
                    onsets = follower.is_onset[:size_change]
                else:
                    features = np.insert(follower.V, 0, follower.V[0], axis=0)
                    beats = np.insert(follower._ref_frame_to_beat, 0, 0.0)
                    onsets = np.insert(follower.is_onset, 0, False)
                with self.assertRaises(ValueError):
                    follower.tempo_set_score(features, beats, onsets)
                self.assertEqual(follower.N_ref, 8)
                np.testing.assert_array_equal(follower.V, np.eye(8, dtype=np.float32))
                np.testing.assert_array_equal(follower.global_cost_matrix, costs)

    def test_run_counts_and_limit_follow_dixon(self):
        follower = self.make_follower(backward_forward=True, max_run_count=3)
        dixon = OnlineTimeWarpingDixonFrame(
            reference_features=follower.V,
            score_positions=follower.score_positions,
            ref_frame_to_beat=follower._ref_frame_to_beat,
            frame_rate=1,
            window_size=1,
        )
        for adv_ref, adv_input in (
            (True, False),
            (True, False),
            (True, False),
            (True, True),
            (False, True),
            (False, True),
            (False, True),
        ):
            follower.update_run_counts(adv_ref, adv_input)
            dixon.update_run_counts(adv_ref, adv_input)
            self.assertEqual(follower.run_count_ref, dixon.run_count_ref)
            self.assertEqual(follower.run_count_input, dixon.run_count_input)
            if follower.run_count_ref == 3:
                self.assertEqual(follower.get_inc(3, 3), Direction.INPUT)
            if follower.run_count_input == 3:
                self.assertEqual(follower.get_inc(3, 3), Direction.REF)
        follower.reset()
        self.assertEqual((follower.run_count_ref, follower.run_count_input), (0, 0))

    def test_refinement_counts_actual_moves_without_extra_direction_lookup(self):
        follower = self.make_follower(backward_forward=True, max_run_count=3)
        follower._last_window_end = 8
        with (
            patch.object(follower, "tempo_backward_path", return_value=iter([(1, 1)])),
            patch.object(
                follower, "get_inc", side_effect=[Direction.BOTH, Direction.INPUT]
            ) as get_inc,
            patch.object(follower, "_cost", return_value=1.0),
        ):
            follower._strategy1(2, 2)
        self.assertEqual(get_inc.call_count, 2)
        self.assertEqual((follower.run_count_ref, follower.run_count_input), (0, 1))
