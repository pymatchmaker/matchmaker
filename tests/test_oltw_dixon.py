#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the dp.oltw_arzt module.
"""

import unittest

import numpy as np
from scipy.spatial import distance as sp_distance

from matchmaker.dp.oltw_dixon import OnlineTimeWarpingDixon

from matchmaker.utils.misc import (
    MatchmakerInvalidOptionError,
    MatchmakerInvalidParameterTypeError,
    RECVQueue
)
from tests.utils import generate_example_sequences

RNG = np.random.RandomState(1984)


class TestOnlineTimeWarpingDixon(unittest.TestCase):
    """
    TT
    """

    def test_init(self):
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

        follower = OnlineTimeWarpingDixon(
            reference_features=X,
            queue=RECVQueue(),
            window_size=1,
            frame_rate=1,
            frame_per_seg=1,
        )

        for i, y in enumerate(Y):
            follower.queue.put((y, i))

    
        for pos in follower.run():
            print(pos)

        print(follower.warping_path)

        print(path)