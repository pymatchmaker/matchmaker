#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Processor related utilities

This module contains all processor related functionality.
"""
from typing import List, Any, Union, Dict, Tuple, Callable


class Processor(object):
    """
    Abstract class for a processor.
    """

    def __call__(
        self,
        data: Any,
        **kwargs: Any,
    ) -> Tuple[Any, Dict[str, Any]]:
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
        kwargs : Dict[str, Any]
            The keyword arguments (to be passed for further processing).
            The method can just pass the input keyword arguments as they are,
            or modify them in any necessary way.
        """

        raise NotImplementedError


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
        self.callable = func

    def __call__(self, data: Any, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        output = self.func(data, **kwargs)

        return output, kwargs


class SequentialOutputProcessor(object):
    """
    Abstract base class for sequential processing of data

    Parameters
    ----------
    processors: list
        List of processors to be applied sequentially.
    """

    processors: List[Processor]

    def __init__(self, processors: Union[Processor, List[Any]]) -> None:
        self.processors = list(processors)

    def __call__(self, data: Any, **kwargs: Any) -> Any:
        """
        Makes a processor callable.
        """
        for proc in self.processors:
            data, kwargs = proc(data, kwargs)
        return data

    def reset(self) -> None:
        """
        Reset the processor. Must be implemented in the derived class
        to reset the processor to its initial state.
        """
        for proc in self.processors:
            if hasattr(proc, "reset"):
                proc.reset()


if __name__ == "__main__":
    pass