#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup for Cython extensions
"""
import numpy as np
from setuptools import Extension, setup

extensions = [
    Extension(
        "matchmaker.utils.distances",
        ["matchmaker/utils/distances.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "matchmaker.dp.dtw_loop",
        ["matchmaker/dp/dtw_loop.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    ext_modules=extensions,
)
