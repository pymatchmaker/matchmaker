#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the dp.oltw_arzt module.
"""
import unittest

import numpy as np

from matchmaker.dp.oltw_arzt import OnlineTimeWarpingArzt
from matchmaker.utils import (CYTHONIZED_METRICS_W_ARGUMENTS,
                              CYTHONIZED_METRICS_WO_ARGUMENTS)
from matchmaker.utils.misc import MatchmakerInvalidOptionError
from tests.utils import generate_example_sequences

RNG = np.random.RandomState(1984)


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
        )

        # Test local_cost_fun as string
        for local_cost_fun in CYTHONIZED_METRICS_WO_ARGUMENTS:

            oltw = OnlineTimeWarpingArzt(
                reference_features=X,
                window_size=2,
                step_size=1,
                local_cost_fun=local_cost_fun,
                start_window_size=2,
            )

            for i, obs in enumerate(Y):
                current_position = oltw(obs)
                # check that the alignments are correct
                self.assertTrue(np.all(path[i] == (current_position, i)))
                # Check that outputs are integers
                self.assertTrue(isinstance(current_position, int))

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
                    )

                    for i, obs in enumerate(Y):
                        current_position = oltw(obs)
                        # check that the alignments are correct
                        self.assertTrue(np.all(path[i] == (current_position, i)))
                        # Check that outputs are integers
                        self.assertTrue(isinstance(current_position, int))
