#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Matchmaker is a library for real-time music alignment
"""

import pkg_resources

__version__ = pkg_resources.get_distribution("matchmaker").version

EXAMPLE_SCORE = pkg_resources.resource_filename(
    "matchmaker",
    "assets/twinkle_twinkle_little_star_score.musicxml",
)

EXAMPLE_PERFORMANCE = pkg_resources.resource_filename(
    "matchmaker",
    "assets/twinkle_twinkle_little_star_performance.mid",
)

EXAMPLE_MATCH = pkg_resources.resource_filename(
    "matchmaker",
    "assets/twinkle_twinkle_little_star_alignment.match",
)
