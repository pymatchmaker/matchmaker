#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Custom exception classes for matchmaker.
"""

from typing import Iterable, List, Union


class MatchmakerInvalidParameterTypeError(Exception):
    """Error for flagging an invalid parameter type."""

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
    """Error for invalid option."""

    def __init__(self, parameter_name, valid_options, value, *args) -> None:
        rqop = ", ".join([f"{op}" for op in valid_options])
        message = f"`{parameter_name}` was expected to be in {rqop}, but is {value}"
        super().__init__(message, *args)


class MatchmakerMissingParameterError(Exception):
    """Error for flagging a missing parameter."""

    def __init__(self, parameter_name: Union[str, List[str]], *args) -> None:
        if isinstance(parameter_name, Iterable) and not isinstance(parameter_name, str):
            message = ", ".join([f"`{pn}`" for pn in parameter_name])
            message = f"{message} were not given"
        else:
            message = f"`{parameter_name}` was not given."
        super().__init__(message, *args)
