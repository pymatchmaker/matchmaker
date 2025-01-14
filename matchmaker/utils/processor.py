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
    """

    def __call__(
        self,
        data: Any,
        **kwargs,
    ) -> Any:
        """
        Parameters
        ----------
        data : Any
            Input data to the processor
        **kwargs: Dict[str, Any]
            Optional keyword arguments

        Returns
        -------
        output: Any
            The output of the processor
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
