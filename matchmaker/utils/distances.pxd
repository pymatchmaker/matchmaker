# -*- coding: utf-8 -*-
# cython language_level 3
cimport numpy

import numpy as np

cimport cython


cdef class Metric:
    cdef float distance(self, float[:] X, float[:] Y) except? 0.0
