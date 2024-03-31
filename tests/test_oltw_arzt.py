#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the dp.oltw_arzt module.
"""
import unittest

import numpy as np
from scipy.spatial import distance as sp_distance

from matchmaker.dp.oltw_arzt import OnlineTimeWarpingArzt
from matchmaker.utils import (
    CYTHONIZED_METRICS_W_ARGUMENTS,
    CYTHONIZED_METRICS_WO_ARGUMENTS,
)
from matchmaker.utils.misc import (
    MatchmakerInvalidOptionError,
    MatchmakerInvalidParameterTypeError,
)
from tests.utils import generate_example_sequences

RNG = np.random.RandomState(1984)

SCIPY_DISTANCES = [
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
    "kulczynski1",
    "rogerstanimoto",
    "russellrao",
    "sokalmichener",
    "sokalsneath",
    "yule",
]


class TestOnlineTimeWarpingArzt(unittest.TestCase):
    def test_local_cost_fun(self):
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

        self.assertTrue(X.dtype == np.float32)
        self.assertTrue(Y.dtype == np.float32)

        # Test raising error if local_cost_fun is invalid type
        self.assertRaises(
            MatchmakerInvalidParameterTypeError,
            OnlineTimeWarpingArzt,
            reference_features=X,
            window_size=2,
            step_size=1,
            # Invalid type (not str, tuple or callable)
            local_cost_fun=RNG.rand(19),
            start_window_size=2,
            frame_rate=1,
        )

        # Test local_cost_fun as string
        for local_cost_fun in CYTHONIZED_METRICS_WO_ARGUMENTS:

            oltw = OnlineTimeWarpingArzt(
                reference_features=X,
                window_size=2,
                step_size=1,
                local_cost_fun=local_cost_fun,
                start_window_size=2,
                frame_rate=1,
            )

            for i, obs in enumerate(Y):
                current_position = oltw(obs)
                # check that the alignments are correct
                self.assertTrue(np.all(path[i] == (current_position, i)))
                # Check that outputs are integers
                self.assertTrue(isinstance(current_position, int))

        # Test that error is raised if incorrect name
        self.assertRaises(
            MatchmakerInvalidOptionError,
            OnlineTimeWarpingArzt,
            reference_features=X,
            window_size=2,
            step_size=1,
            local_cost_fun="wrong_local_cost_fun",
            start_window_size=2,
        )

        # Test local_cost_fun as tuple
        for local_cost_fun in CYTHONIZED_METRICS_W_ARGUMENTS:

            if local_cost_fun == "Lp":
                for p in RNG.uniform(low=1, high=10, size=10):
                    oltw = OnlineTimeWarpingArzt(
                        reference_features=X,
                        window_size=2,
                        step_size=1,
                        local_cost_fun=(local_cost_fun, dict(p=p)),
                        start_window_size=2,
                        frame_rate=1,
                    )

                    for i, obs in enumerate(Y):
                        current_position = oltw(obs)
                        # check that the alignments are correct
                        self.assertTrue(np.all(path[i] == (current_position, i)))
                        # Check that outputs are integers
                        self.assertTrue(isinstance(current_position, int))

        # Test that error is raised if incorrect name
        self.assertRaises(
            MatchmakerInvalidOptionError,
            OnlineTimeWarpingArzt,
            reference_features=X,
            window_size=2,
            step_size=1,
            local_cost_fun=("wrong_local_cost_fun", {"param": "value"}),
            start_window_size=2,
            frame_rate=1,
        )

        for spdist in SCIPY_DISTANCES:

            oltw = OnlineTimeWarpingArzt(
                reference_features=X,
                window_size=2,
                step_size=1,
                local_cost_fun=getattr(sp_distance, spdist),
                start_window_size=2,
                frame_rate=1,
            )

            for i, obs in enumerate(Y):
                current_position = oltw(obs)
                # with some of the scipy metrics, we cannot
                # ensure that the results will always
                # be correct, so we only
                # check if the output types are correct
                self.assertTrue(isinstance(current_position, int))
