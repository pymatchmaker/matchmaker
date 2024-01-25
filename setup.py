#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np

# Package meta-data.
NAME = "matchmaker"
DESCRIPTION = "A package for real-time music alignment"
KEYWORDS = "music alignment midi audio"
URL = "https://github.com/neosatrapahereje/matchmaker"
EMAIL = "carloscancinochacon@gmail.com"
AUTHOR = "Carlos Cancino-Chacón, Jiyun Park"
REQUIRES_PYTHON = ">=3.9"
VERSION = "0.0.1"

# What packages are required for this module to be executed?
REQUIRED = ["python-rtmidi", "mido", "cython", "numpy", "scipy", "madmom", "partitura"]

# What packages are optional?
EXTRAS = {
    # 'fancy feature': ['django'],
}

SCRIPTS = [
    "bin/Audio2AudioAlignment",
]

include_dirs = [np.get_include()]

extensions = [
    Extension(
        "matchmaker.utils.distances",
        ["matchmaker/utils/distances.pyx"],
        include_dirs=include_dirs,
        # extra_compile_args=['-fopenmp'],
        # extra_link_args=['-fopenmp'],
    ),
    Extension(
        "matchmaker.alignment.offline.dtw",
        ["matchmaker/alignment/offline/dtw.pyx"],
        include_dirs=include_dirs,
        # extra_compile_args=['-fopenmp'],
        # extra_link_args=['-fopenmp'],
    ),
]


# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

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
    ext_modules=cythonize(extensions, language_level=3),
    install_requires=REQUIRED,
    scripts=SCRIPTS,
    # extras_require=EXTRAS,
    # include_package_data=True,
    license="Apache 2.0",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: MIDI",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
