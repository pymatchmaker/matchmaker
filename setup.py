#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os

import numpy as np
from setuptools import Extension, find_packages, setup

# from distutils.extension import Extension

# from Cython.Build import cythonize


# Package meta-data.
NAME = "matchmaker"
DESCRIPTION = "A package for real-time music alignment"
KEYWORDS = "music alignment midi audio"
URL = "https://github.com/neosatrapahereje/matchmaker"
EMAIL = "carloscancinochacon@gmail.com"
AUTHOR = "Matchmaker Development Team"
REQUIRES_PYTHON = ">=3.9"
VERSION = "0.0.1"

SETUP_REQUIRES = [
    # Setuptools 18.0 properly handles Cython extensions.
    "setuptools>=18.0",
    "cython>=3.0.0",
    "numpy>=1.26.0",
]

# What packages are required for this module to be executed?
REQUIRED = [
    "cython>=3.0.0",
    "python-rtmidi>=1.5.8",
    "mido>=1.3.0",
    "numpy>=1.26.0",
    "scipy>=1.11.3",
    "librosa>=0.10.1",
    # For now, madmom needs to be installed directly from GitHub
    "partitura>=1.3.0",
    "madmom>=0.17.dev0",
    "python-hiddenmarkov>=0.1.3",
    "pyaudio>=0.2.14",
]


EXTRAS = {
    # 'fancy feature': ['django'],
}

SCRIPTS = []

include_dirs = [np.get_include()]

extensions = [
    Extension(
        "matchmaker.utils.distances",
        ["matchmaker/utils/distances.pyx"],
        include_dirs=include_dirs,
    ),
    Extension(
        "matchmaker.dp.dtw_loop",
        ["matchmaker/dp/dtw_loop.pyx"],
        include_dirs=include_dirs,
    ),
]

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=KEYWORDS,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    # ext_modules=cythonize(extensions, language_level=3),
    ext_modules=extensions,
    install_requires=REQUIRED,
    setup_requires=SETUP_REQUIRES,
    scripts=SCRIPTS,
    # extras_require=EXTRAS,
    # include_package_data=True,
    license="Apache 2.0",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: MIDI",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
