"""
Strategy 10: Global cosine curve fit (harmonic regression peak).

Fits a single-cycle cosine function to the "goodness" curve (goodness =
higher-is-better, inverse of novelty) via least squares over ALL samples,
and takes the peak location of the fitted continuous curve as the optimal
degree. This is more principled than a plain weighted average because it
explicitly models a peaked (unimodal) shape instead of just averaging, so
it is less biased by an asymmetric tail of the sampled data.

Model: goodness(theta) ~= A + C*cos(theta) + S*sin(theta)
     = A + B*cos(theta - phi)   (B = sqrt(C^2 + S^2), phi = atan2(S, C))

Standard library + numpy only -- no dependency on the insect_nav package,
so this module can be loaded standalone via importlib (some of the host
environment's insect_nav dependencies, e.g. tqdm/scipy, may be missing).
"""
import math

import numpy as np


def select_degree(degree_array: list, novelty_array: list, step: float, prev_degree=None) -> tuple:
    """Returns (optimal_degree, uncertainty)."""
    degrees = np.asarray(degree_array, dtype=np.float64)
    novelty = np.asarray(novelty_array, dtype=np.float64)

    goodness = np.max(novelty) - novelty
    theta = np.radians(degrees)

    X = np.column_stack((np.ones_like(theta), np.cos(theta), np.sin(theta)))
    y = goodness

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    A, C, S = coeffs

    optimal_degree = math.degrees(math.atan2(S, C))

    amplitude_b = math.sqrt(C ** 2 + S ** 2)
    residuals = y - X @ coeffs
    rms_residual = math.sqrt(np.mean(residuals ** 2))
    uncertainty = rms_residual / (amplitude_b + rms_residual + 1e-9)

    return optimal_degree, uncertainty
