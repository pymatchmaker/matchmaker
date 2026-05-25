#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Matchmaker is a library for real-time music alignment
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import dp, features, io, prob, utils
from .matchmaker import *

__all__ = ["dp", "features", "io", "prob", "utils"]

try:
    __version__ = version("pymatchmaker")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.3.0"

ASSETS_DIR = Path(__file__).parent / "assets"

EXAMPLE_PIECES = {
    "simple_mozart": {
        "score": str(ASSETS_DIR / "simple_mozart_k265_var1.musicxml"),
        "midi": str(ASSETS_DIR / "simple_mozart_k265_var1.mid"),
        "audio": str(ASSETS_DIR / "simple_mozart_k265_var1.mp3"),
        "match": str(ASSETS_DIR / "simple_mozart_k265_var1.match"),
        "annotations": str(ASSETS_DIR / "simple_mozart_k265_var1_note_annotations.txt"),
    },
    "bach_fugue": {
        "score": str(ASSETS_DIR / "Bach-fugue_bwv_858.musicxml"),
        "midi": str(ASSETS_DIR / "Bach-fugue_bwv_858.mid"),
        "audio": str(ASSETS_DIR / "Bach-fugue_bwv_858.mp3"),
        "match": str(ASSETS_DIR / "Bach-fugue_bwv_858.match"),
        "annotations": str(ASSETS_DIR / "Bach-fugue_bwv_858_note_annotations.txt"),
        "beat_annotations": str(ASSETS_DIR / "Bach-fugue_bwv_858_beat_annotations.txt"),
    },
}

# Legacy constants
EXAMPLE_SCORE = str(ASSETS_DIR / "mozart_k265_var1.musicxml")
EXAMPLE_PERFORMANCE = str(ASSETS_DIR / "mozart_k265_var1.mid")
EXAMPLE_MATCH = str(ASSETS_DIR / "mozart_k265_var1.match")
EXAMPLE_AUDIO = str(ASSETS_DIR / "mozart_k265_var1.mp3")
