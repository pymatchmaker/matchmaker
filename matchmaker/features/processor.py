#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Processor related utilities

This module contains all processor related functionality.
"""

from typing import Any, Callable


class Processor(object):
    """
    Abstract class for a processor.

    Sits between a Stream and the queue feeding the score follower.

    Input
    -----
    A ``(data, frame_time)`` tuple where:
      - ``data`` is ``np.ndarray`` for audio buffers, or
        ``List[Tuple[mido.Message, m_time]]`` for MIDI messages.
      - ``frame_time`` (float) is the stream's nominal time for the frame.

    Output
    ------
    Either a ``(features, perf_time)`` tuple, or ``None`` while buffering
    (e.g. a chord-buffering processor still waiting for the next note).
    Most processors pass ``frame_time`` through as ``perf_time``;
    chord-buffering MIDI processors override it with the chord onset
    time (the first note's ``m_time`` of the buffered chord).

    The Stream forwards the returned tuple to the queue unchanged, so
    downstream code (`OnlineAlignment.__call__`) always sees
    ``(features, perf_time)``.
    """

    def __call__(
        self,
        data: Any,
        **kwargs,
    ) -> Any:
        """
        Parameters
        ----------
        data : Tuple[Any, float]
            ``(data, frame_time)`` tuple from the Stream.

        Returns
        -------
        output : Tuple[np.ndarray, float] or None
            ``(features, perf_time)`` or ``None`` while buffering.
        """

        raise NotImplementedError

    def reset(self):
        """
        Resets the processor, if it has any internal states.

        This method needs to be implemented in derived classes if needed.
        """
        pass


class ProcessorWrapper(Processor):
    """
    Wraps a function as a Processor class

    Parameters
    ----------
    func : Callable
        Function to be wrapped as a `Processor`.

    Attributes
    ----------
    func : Callable
        Function wrapped as a processor.
    """

    func: Callable[[Any], Any]

    def __init__(self, func: Callable[[Any], Any]) -> None:
        super().__init__()
        self.func = func

    def __call__(self, data: Any, **kwargs) -> Any:
        output = self.func(data, **kwargs)

        return output


class DummyProcessor(Processor):
    """
    Dummy sequential output processor, which always returns
    the inputs unmodified inputs
    """

    def __call__(
        self,
        data: Any,
        **kwargs,
    ) -> Any:
        return data


if __name__ == "__main__":
    pass
