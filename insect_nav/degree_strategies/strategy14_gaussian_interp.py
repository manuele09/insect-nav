"""
Strategy 14: Gaussian sub-bin interpolation.

Same idea as strategy 3 (parabolic sub-bin interpolation around the global
minimum) but using the "Gaussian interpolation" formula standard in
cross-correlation / pitch-detection peak fitting: fit a parabola to the LOG
of the (inverted, i.e. higher-is-better) values around the extremum instead
of to the raw values. This is the correct closed-form peak estimator when
the true underlying curve is locally Gaussian-shaped rather than
quadratic -- literature on sub-pixel disparity/registration (see e.g.
Pallotta et al., IEEE TGRS 2019) reports Gaussian interpolation often
outperforming plain parabolic fitting for peaks that aren't exactly
quadratic.

Standalone module: only the Python standard library (math) is used, so it
can be loaded directly via importlib.util.spec_from_file_location without
importing the insect_nav package.
"""
import math


def select_degree(degree_array: list, novelty_array: list, step: float, prev_degree=None) -> tuple:
    """Returns (optimal_degree, uncertainty)."""
    n = len(novelty_array)
    min_value = min(novelty_array)
    max_value = max(novelty_array)

    tied_idx = [i for i, v in enumerate(novelty_array) if v == min_value]
    idx = min(tied_idx, key=lambda i: abs(degree_array[i]))  # among ties, closest to 0 degrees

    # Convert novelty (lower=better) to a strictly-positive "goodness" so we
    # can take logs: goodness = (max - novelty) + 1, guarantees >= 1.
    def goodness(i):
        return (max_value - novelty_array[i]) + 1.0

    g0 = goodness(idx)
    g_m1 = goodness(idx - 1) if idx - 1 >= 0 else g0
    g_p1 = goodness(idx + 1) if idx + 1 < n else g0

    log_m1, log_0, log_p1 = math.log(g_m1), math.log(g0), math.log(g_p1)
    denom = log_m1 - 2 * log_0 + log_p1
    delta = 0.5 * (log_m1 - log_p1) / denom if denom != 0 else 0.0
    delta = max(-0.5, min(0.5, delta))

    optimal_degree = degree_array[idx] + delta * step

    uncertainty = 1 - (1 / len(tied_idx))  # same convention as strategy 3

    return optimal_degree, uncertainty
