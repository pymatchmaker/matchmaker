# -*- coding: utf-8 -*-
# cython language_level 3
"""
Cythonized methods for computing distances
"""
cimport numpy as np

import numpy as np

cimport cython
from libc.math cimport abs, sqrt


cdef class Metric:
    """
    Base class for defining a metric in Cython
    """
    def __call__(self, float[:] X, float[:] Y):
        return self.distance(X, Y)
    cdef float distance(self, float[:] X, float[:] Y) except? 0.0:
        raise NotImplementedError()


@cython.boundscheck(False)
@cython.wraparound(False)
def cdist(float[:, :]  X, float[:, :] Y, Metric local_distance):
    """
    Pairwise distance between the elements of two arrays

    Parameters
    ----------
    X : float np.ndarray
        2D array with shape (M, L), where M is the number of
        L-dimensional elements in X.
    Y : float np.ndarray
        2D array with shapes (N, L), where N is the number of
        L-dimensional elements in Y.
    local_distance : callable
        Callable function for computing the local distance between the
        elements of X and Y. See e.g., metrics defined below (`Euclidean`,
        `Cosine`, etc.)

    Returns
    -------
    D : np.ndarray
        A 2D array where the i,j element represents the distance
        from the i-th element of X to the j-th element of Y according
        to the specified metric (in `local_distance`).
    """
    # Initialize variables
    cdef Py_ssize_t M = X.shape[0]
    cdef Py_ssize_t N = Y.shape[0]
    cdef float[:, :] D = np.empty((M, N), dtype=np.float32)
    cdef Py_ssize_t i, j

    # Loop for computing the distance between each element
    # for i in prange(M, nogil=True):
    for i in range(M):
        for j in range(N):
            D[i, j] = local_distance.distance(X[i], Y[j])
    return np.asarray(D)


@cython.boundscheck(False)
@cython.wraparound(False)
def vdist(float[:, :]  X, float[:] Y, Metric local_distance):
    # Initialize variables
    cdef Py_ssize_t M = X.shape[0]
    cdef float[:] D = np.empty(M, dtype=np.float32)
    cdef Py_ssize_t i

    # Loop for computing the distance between each element
    for i in range(M):
        D[i] = local_distance.distance(X[i], Y)
    return np.asarray(D)


@cython.boundscheck(False)
@cython.boundscheck(False)
cdef class Euclidean(Metric):
    cdef float distance(self, float[:] X, float[:] Y) except? 0.0:
        """
        Euclidean Distance between vectors

        Parameters
        ----------
        X : float np.ndarray
            An M dimensional vector
        Y : float np.ndarray
            An M dimensional vector

        Returns
        -------
        dist : float
            The distance between X and Y
        """
        cdef Py_ssize_t M = X.shape[0]
        cdef float diff, dist
        cdef Py_ssize_t i

        dist = 0.0
        for i in range(M):
            diff = X[i] - Y[i]
            dist += diff * diff
        return sqrt(dist)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef class Cosine(Metric):
    cdef float distance(self, float[:] X, float[:] Y) except? 0.0:
        """
        Cosine Distance between vectors

        Parameters
        ----------
        X : float np.ndarray
            An M dimensional vector
        Y : float np.ndarray
            An M dimensional vector

        Returns
        -------
        dist : float
            The distance between X and Y
        """
        cdef Py_ssize_t M = X.shape[0]
        cdef float dot = 0, norm_x = 0, norm_y = 0
        cdef float cos, dist
        cdef float eps = 1e-10
        cdef Py_ssize_t i

        for i in range(M):
            dot += (X[i] * Y[i])
            norm_x += X[i] ** 2
            norm_y += Y[i] ** 2

        cos = dot / (sqrt(norm_x) * sqrt(norm_y) + eps)

        dist = 1 - cos

        return dist


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class L1(Metric):
    cdef float distance(self, float[:] X, float[:] Y) except? 0.0:
        """
        L1- norm between vectors

        Parameters
        ----------
        X : float np.ndarray
            An M dimensional vector
        Y : float np.ndarray
            An M dimensional vector

        Returns
        -------
        dist : float
            The distance between X and Y
        """
        cdef Py_ssize_t M = X.shape[0]
        cdef float diff
        cdef float dist = 0
        cdef Py_ssize_t i

        for i in range(M):
            diff = X[i] - Y[i]
            dist += abs(diff)
        return dist

# Alias
Manhattan = L1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef class Lp(Metric):
    cdef float p
    cdef float pinv
    def __init__(self, float p):
        self.p = p
        self.pinv = 1. / (p + 1e-10)

    cdef float distance(self, float[:] X, float[:] Y) except? 0.0:
        """
        Lp - metric
        """
        cdef Py_ssize_t M = X.shape[0]
        cdef float dist = 0.0
        cdef float diff
        cdef Py_ssize_t i

        for i in range(M):
            diff = abs(X[i] - Y[i])
            dist += diff ** self.p

        return dist ** self.pinv

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class Linf(Metric):
    cdef float distance(self, float[:] X, float[:] Y) except? 0.0:
        """
        L_inf- norm between vectors

        Parameters
        ----------
        X : float np.ndarray
            An M dimensional vector
        Y : float np.ndarray
            An M dimensional vector

        Returns
        -------
        dist : float
            The distance between X and Y
        """
        cdef Py_ssize_t M = X.shape[0]
        cdef float[:] diff = np.zeros(M, dtype=np.float32)
        cdef Py_ssize_t i

        for i in range(M):
            diff[i] = abs(X[i] - Y[i])
        return max(diff)
