"""
Strategy 1: Softmax-weighted centroid.

Instead of picking the global-minimum-novelty shift(s) like the production
`find_optimal_degree`, this strategy uses the ENTIRE novelty curve: it
converts novelty into a softmax weight (lower novelty -> higher weight) and
returns the weighted arithmetic mean of the degree_array as the optimal
heading. Uncertainty is reported as the normalized Shannon entropy of the
weight distribution.

Standalone module: only the Python standard library and numpy are used, so
it can be loaded directly via importlib.util.spec_from_file_location without
importing the (partially broken) insect_nav package.
"""
import math

import numpy as np

T = 1.0  # softmax temperature hyperparameter


def select_degree(degree_array: list, novelty_array: list, step: float, prev_degree=None) -> tuple:
    """Returns (optimal_degree, uncertainty)."""
    degrees = np.asarray(degree_array, dtype=float)
    novelty = np.asarray(novelty_array, dtype=float)

    weights = np.exp(-novelty / T)
    weight_sum = weights.sum()

    optimal_degree = float(np.sum(weights * degrees) / weight_sum)

    p = weights / weight_sum
    # treat 0 * log(0) as 0
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    entropy = float(-np.sum(terms))
    max_entropy = math.log(len(degree_array))
    uncertainty = entropy / max_entropy

    return optimal_degree, uncertainty
