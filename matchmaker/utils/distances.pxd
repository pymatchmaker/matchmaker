# -*- coding: utf-8 -*-
# cython language_level 3
cimport numpy

import numpy as np

cimport cython


cdef class Metric:
    cdef double distance(self, double[:] X, double[:] Y) except? 0.0
