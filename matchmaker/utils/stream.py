#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains all Stream related functionality.
"""

from __future__ import annotations

import time
from threading import Thread
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable, Optional, Type, Union

if TYPE_CHECKING:  # pragma: no cover
    from matchmaker.utils.processor import Processor


class Stream(Thread):
    """Abstract class for Streams.

    Parameters
    ----------
    processor : Union[Callable, Processor]
        A `Processor` instance or a callable for extracting
        features from the inputs.
    mock : bool
        A boolean indicating whether to run the stream offline
        given a file as an input instead of real-time inputs.
    """

    mock: bool
    init_time: Optional[float]
    processor: Union[Callable, Processor]
    listen: bool

    def __init__(
        self,
        processor: Union[Callable, Processor],
        mock: bool,
    ) -> None:
        Thread.__init__(self)
        self.processor = processor
        self.mock = mock
        self.listen = False
        self.init_time = None

    def start_listening(self):
        """
        Set listen to True and sets initial time
        """
        self.listen = True

        if self.mock:
            # start from zero if running the stream
            # in offline mode
            self.init_time = 0.0
        else:
            self.init_time = time.time()

    def stop_listening(self):
        """Set listen to false"""
        self.listen = False

    def _process_frame(self, data: Any, *args, **kwargs) -> Any:
        """Process incoming frame.

        This method should be implemented in derived subclasses.

        Parameters
        ----------
        data : Any
            Incoming data.
        """
        raise NotImplementedError

    def mock_stream(self):
        """
        Mock stream offline

        This method should be implemented in derived subclasses
        """
        raise NotImplementedError

    def __enter__(self):
        """Enter method for context managers.

        This method should be implemented in derived subclasses.
        """
        raise NotImplementedError

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:  # pragma: no cover
        """Enter method for context managers.

        This method should be implemented in derived subclasses.
        """
        raise NotImplementedError

    def run(self):
        """Run thread method

        This method should be implemented in derived subclasses.
        """
        raise NotImplementedError

    def stop(self):
        """Enter method for context managers.

        This method should be implemented in derived subclasses.
        """
        raise NotImplementedError
