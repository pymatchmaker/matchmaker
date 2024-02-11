#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Miscellaneous utilities
"""
import numbers
from typing import Union
import numpy as np


def ensure_rng(
    seed: Union[numbers.Integral, np.random.RandomState]
) -> np.random.RandomState:
    """
    Ensure random seed generator

    Parameters
    ----------
    seed : int or np.random.RandomState
        An integer to serve as the seed for the random number generator or a
        `np.random.RandomState` instance.

    Returns
    -------
    rng : np.random.RandomState
        A random number generator.
    """

    if isinstance(seed, numbers.Integral):
        rng = np.random.RandomState(seed)
        return rng
    elif isinstance(seed, np.random.RandomState):
        rng = seed
        return rng
    else:
        raise ValueError(
            "`seed` should be an integer or an instance of "
            f"`np.random.RandomState` but is {type(seed)}"
        )
