#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Miscellaneous utilities
"""
import numbers
from queue import Empty, Queue
from typing import Any, Iterable, Union

import numpy as np


class MatchmakerInvalidParameterTypeError(Exception):
    """
    Error for flagging an invalid parameter type.
    """

    def __init__(
        self,
        parameter_name: str,
        required_parameter_type: Union[type, Iterable[type]],
        actual_parameter_type: type,
        *args,
    ) -> None:

        if isinstance(required_parameter_type, Iterable):
            rqpt = ", ".join([f"{pt}" for pt in required_parameter_type])
        else:
            rqpt = required_parameter_type
        message = f"`{parameter_name}` was expected to be {rqpt}, but is {actual_parameter_type}"

        super().__init__(message, *args)


class MatchmakerInvalidOptionError(Exception):
    """
    Error for invalid option.
    """

    def __init__(self, parameter_name, valid_options, value, *args) -> None:

        rqop = ", ".join([f"{op}" for op in valid_options])
        message = f"`{parameter_name}` was expected to be in {rqop}, but is {value}"

        super().__init__(message, *args)


def ensure_rng(
    seed: Union[numbers.Integral, np.random.RandomState]
) -> np.random.RandomState:
    """
    Ensure random number generator is a np.random.RandomState instance

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


class RECVQueue(Queue):
    """
    Queue with a recv method (like Pipe)

    This class uses python's Queue.get with a timeout makes it interruptable via KeyboardInterrupt
    and even for the future where that is possibly out-dated, the interrupt can happen after each timeout
    so periodically query the queue with a timeout of 1s each attempt, finding a middleground
    between busy-waiting and uninterruptable blocked waiting
    """

    def __init__(self) -> None:
        Queue.__init__(self)

    def recv(self) -> Any:
        """
        Return and remove an item from the queue.
        """
        while True:
            try:
                return self.get(timeout=1)
            except Empty:
                pass

    def poll(self) -> bool:
        return self.empty()
