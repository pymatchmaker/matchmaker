#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Stream related utilities

This module contains all Stream related functionality.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Union, Any

if TYPE_CHECKING:
    from matchmaker.utils.processor import Processor



class Stream(object):
    """
    Abstract class for a stream
    """

    def __init__(self, processor: Union[Callable, Processor]) -> None:
        self.processor = processor

    def _process_feature(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def _process_frame(self, data: Any, *args, **kwargs) -> Any:
        raise NotImplementedError