#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the dp.oltw_arzt module.
"""

import unittest

import numpy as np
from scipy.spatial import distance as sp_distance

from matchmaker.dp.oltw_arzt import OnlineTimeWarpingArztFrame as OnlineTimeWarpingArzt
from matchmaker.utils import (
    CYTHONIZED_METRICS_W_ARGUMENTS,
    CYTHONIZED_METRICS_WO_ARGUMENTS,
)
from matchmaker.utils.errors import (
    MatchmakerInvalidOptionError,
    MatchmakerInvalidParameterTypeError,
)
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

            for i, obs in enumerate(Y):
                current_position = oltw(obs, float(i))
                # with some of the scipy metrics, we cannot
                # ensure that the results will always
                # be correct, so we only
                # check if the output types are correct
                self.assertTrue(isinstance(current_position, float))
