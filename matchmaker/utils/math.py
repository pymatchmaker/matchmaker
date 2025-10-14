#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This contains math-related utilities
"""
from typing import Union
import numpy as np
EPS = 1e-6

def cauchy_density(
    x: Union[np.ndarray, float],
    median: Union[np.ndarray, float],
    width: float,
    log: bool = False,
):
    """
    Cauchy distribution
    """
    pdf = 1.0 / (np.pi * (1 + ((x - median) / width) ** 2))
    if log:
        return np.log(pdf)
    return pdf


def normal_density(x, mu, sigma2, log=False):
    sigma2 = max([float(sigma2), EPS])
    log_pdf = - .5 * (np.log(2 * np.pi * sigma2) + (x - mu) ** 2 / sigma2)
    if log:
        return log_pdf
    return np.exp(log_pdf)



def discrete_normal_density(x, mu, sigma2, grid, log=False):
    # return multivariate_normal.pdf(x, mu, sigma2)
    x = np.atleast_1d(x).astype(float)
    if sigma2 == 0:
        out = (x == mu).astype(float)
    else:
        # evaluate the normal density on the grid
        prob = normal_density(grid, mu, sigma2)
        # normalize to have a discrete distribution
        out = normal_density(x, mu, sigma2) * np.in1d(x, grid) / np.sum(prob)
    if log:
        out = np.log(out)
    return out
