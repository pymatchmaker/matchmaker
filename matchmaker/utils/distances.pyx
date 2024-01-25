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
    def __call__(self, double[:] X, double[:] Y):
        return self.distance(X, Y)
    cdef double distance(self, double[:] X, double[:] Y) except? 0.0:
        raise NotImplementedError()


@cython.boundscheck(False)
@cython.wraparound(False)
def cdist(double[:, :]  X, double[:, :] Y, Metric local_distance):
    """
    Pairwise distance between the elements of two arrays
    
    Parameters
    ----------
    X : double np.ndarray
        2D array with shape (M, L), where M is the number of
        L-dimensional elements in X.
    Y : double np.ndarray
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
    cdef double[:, :] D = np.empty((M, N), dtype=float)
    cdef Py_ssize_t i, j

    # Loop for computing the distance between each element
    # for i in prange(M, nogil=True):
    for i in range(M):
        for j in range(N):
            D[i, j] = local_distance.distance(X[i], Y[j])
    return np.asarray(D)


@cython.boundscheck(False)
@cython.wraparound(False)
def vdist(double[:, :]  X, double[:] Y, Metric local_distance):
    # Initialize variables
    cdef Py_ssize_t M = X.shape[0]
    cdef double[:] D = np.empty(M, dtype=float)
    cdef Py_ssize_t i

    # Loop for computing the distance between each element
    for i in range(M):
        D[i] = local_distance.distance(X[i], Y)
    return np.asarray(D)
    

@cython.boundscheck(False)
@cython.boundscheck(False)
cdef class Euclidean(Metric):
    cdef double distance(self, double[:] X, double[:] Y) except? 0.0:
        """
        Euclidean Distance between vectors

        Parameters
        ----------
        X : double np.ndarray
            An M dimensional vector
        Y : double np.ndarray
            An M dimensional vector

        Returns
        -------
        dist : double
            The distance between X and Y
        """
        cdef Py_ssize_t M = X.shape[0]
        cdef double diff, dist
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
    cdef double distance(self, double[:] X, double[:] Y) except? 0.0:
        """
        Cosine Distance between vectors

        Parameters
        ----------
        X : double np.ndarray
            An M dimensional vector
        Y : double np.ndarray
            An M dimensional vector

        Returns
        -------
        dist : double
            The distance between X and Y
        """
        cdef Py_ssize_t M = X.shape[0]
        cdef double dot = 0, norm_x = 0, norm_y = 0
        cdef double cos, dist
        cdef double eps = 1e-10
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
    cdef double distance(self, double[:] X, double[:] Y) except? 0.0:
        """
        L1- norm between vectors

        Parameters
        ----------
        X : double np.ndarray
            An M dimensional vector
        Y : double np.ndarray
            An M dimensional vector

        Returns
        -------
        dist : double
            The distance between X and Y
        """
        cdef Py_ssize_t M = X.shape[0]
        cdef double diff
        cdef double dist = 0
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
    cdef double p
    cdef double pinv
    def __init__(self, double p):
        self.p = p
        self.pinv = 1. / (p + 1e-10)

    cdef double distance(self, double[:] X, double[:] Y) except? 0.0:
        """
        Lp - metric
        """
        cdef Py_ssize_t M = X.shape[0]
        cdef double dist = 0.0
        cdef double diff
        cdef Py_ssize_t i

        for i in range(M):
            diff = abs(X[i] - Y[i])
            dist += diff ** self.p

        return dist ** self.pinv

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class Linf(Metric):
    cdef double distance(self, double[:] X, double[:] Y) except? 0.0:
        """
        L_inf- norm between vectors

        Parameters
        ----------
        X : double np.ndarray
            An M dimensional vector
        Y : double np.ndarray
            An M dimensional vector

        Returns
        -------
        dist : double
            The distance between X and Y
        """
        cdef Py_ssize_t M = X.shape[0]
        cdef double[:] diff = np.zeros(M, dtype=float)
        cdef Py_ssize_t i

        for i in range(M):
            diff[i] = abs(X[i] - Y[i])
        return max(diff)
