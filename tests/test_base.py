#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the base module.
"""
import unittest

import numpy as np

from matchmaker.base import OnlineAlignment
from tests.utils import generate_example_sequences

RNG = np.random.RandomState(1984)


class TestOnlineAlignment(unittest.TestCase):
    def test_raise_not_implemented(self):

        X, Y, path = generate_example_sequences(
            lenX=10,
            centers=3,
            n_features=3,
            maxreps=4,
            minreps=1,
            noise_scale=0.00,
            random_state=RNG,
        )
        follower = OnlineAlignment(reference_features=X)

        for obs in Y:
            self.assertRaises(NotImplementedError, follower, obs)
