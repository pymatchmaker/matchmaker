"""Shared reference/input directions for online time warping."""

from enum import IntEnum


class Direction(IntEnum):
    """Axes advanced by a step; NONE marks a missing predecessor."""

    NONE = -1
    REF = 0
    INPUT = 1
    BOTH = 2
