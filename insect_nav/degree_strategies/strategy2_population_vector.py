"""
Strategy 2: Population vector (circular weighted average) decoding.

Inspired by population coding in biological compass/heading neurons: each
scanned angular shift "votes" for its own direction with a weight inversely
proportional to its novelty score (lower novelty = more familiar = higher
weight). The optimal heading is the direction of the resultant weighted
vector sum, and the uncertainty is derived from how concentrated (vs.
spread out/conflicting) the population's votes are.

Standalone module: only stdlib + numpy, no insect_nav imports, so it can be
loaded directly via importlib without pulling in insect_nav's dependency
chain.
"""
import math

import numpy as np


def select_degree(degree_array: list, novelty_array: list, step: float, prev_degree=None) -> tuple:
    """Returns (optimal_degree, uncertainty)."""
    novelty = np.asarray(novelty_array, dtype=np.float64)
    degrees_arr = np.asarray(degree_array, dtype=np.float64)

    max_n = np.max(novelty)
    weights = max_n - novelty  # inverse-linear weight, >= 0

    weight_sum = np.sum(weights)
    if weight_sum == 0:
        weights = np.ones_like(weights)
        weight_sum = np.sum(weights)

    radians = np.radians(degrees_arr)
    Vx = np.sum(weights * np.cos(radians))
    Vy = np.sum(weights * np.sin(radians))

    optimal_degree = float(math.degrees(math.atan2(Vy, Vx)))

    R = math.sqrt(Vx ** 2 + Vy ** 2) / weight_sum
    uncertainty = 1.0 - R

    return optimal_degree, uncertainty
