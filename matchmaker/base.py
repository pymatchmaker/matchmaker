# -*- coding: utf-8 -*-
from typing import Any


class OnlineAlignment(object):
    """
    Base class for online alignment methods.

    Parameters
    ----------
    reference_features : Any
        Features of the music we want to align our online input to.
    """

    def __init__(self, reference_features: Any) -> None:
        self.reference_features: Any = reference_features

    def __call__(self, input: Any) -> float:
        """
        Return the current position
        """
        raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover
    pass
