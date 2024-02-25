#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Matchmaker is a library for real-time music alignment
"""

import pkg_resources

__version__ = pkg_resources.get_distribution("matchmaker").version

EXAMPLE_SCORE = pkg_resources.resource_filename(
    "matchmaker",
    "assets/mozart_k265_var1.musicxml",
)

EXAMPLE_PERFORMANCE = pkg_resources.resource_filename(
    "matchmaker",
    "assets/mozart_k265_var1.mid",
)

EXAMPLE_MATCH = pkg_resources.resource_filename(
    "matchmaker",
    "assets/mozart_k265_var1.match",
)

EXAMPLE_MATCH = pkg_resources.resource_filename(
    "matchmaker",
    "assets/mozart_k265_var1.mp3",
)