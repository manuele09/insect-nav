"""
Strategy 6: triangular smoothing + the exact same grouping/argmin logic
used by `find_optimal_degree` in insect_nav/base.py.

This isolates the effect of noise-suppression (smoothing the novelty
curve before selection) from the rest of the already-validated
grouping/tie-breaking logic, which is copied here verbatim (as a
private helper) and left otherwise unmodified except for using
`np.isclose` instead of strict equality when locating the minimum,
since smoothed values are floats.

Standalone module: standard library + numpy only. Do NOT import
anything from the `insect_nav` package (this must be loadable via
importlib.util.spec_from_file_location in environments missing some
of insect_nav's dependencies).
"""
import numpy as np


def _smooth_triangular(novelty_array):
    """Apply the [0.25, 0.5, 0.25] triangular kernel to novelty_array with
    edge-clamped (replicated) boundaries -- NOT circular, since the domain
    is a straight +/-90 degree arc, not a full circle."""
    n = len(novelty_array)
    smoothed = []
    for i in range(n):
        left = novelty_array[i - 1] if i - 1 >= 0 else novelty_array[0]
        right = novelty_array[i + 1] if i + 1 < n else novelty_array[n - 1]
        center = novelty_array[i]
        smoothed.append(0.25 * left + 0.5 * center + 0.25 * right)
    return smoothed


def _grouping_argmin(degree_array, values, step):
    """Verbatim copy of the grouping/tie-breaking logic from
    insect_nav.base.find_optimal_degree, applied to `values` (which may be
    the raw novelty_array or a smoothed float version of it). Uses
    np.isclose instead of strict equality when locating the minimum, since
    floating point equality would be too strict for smoothed values."""
    min_value = min(values)
    deg_min = [d for d, v in zip(degree_array, values) if np.isclose(v, min_value)]

    groups, current = [], [deg_min[0]]
    for d in deg_min[1:]:
        if abs(d - current[-1]) == step:
            current.append(d)
        else:
            groups.append(current)
            current = [d]
    groups.append(current)

    max_length = max(len(g) for g in groups)
    longest = [g for g in groups if len(g) == max_length]
    best_group = longest[0] if len(longest) == 1 else min(longest, key=lambda g: abs(np.mean(g)))

    optimal_degree = float(np.mean(best_group))
    indecision_a = 1 - len(best_group) / len(deg_min)
    indecision_b = (len(best_group) - 1) / len(degree_array)
    uncertainty = (indecision_a + indecision_b) / 2

    return optimal_degree, uncertainty


def select_degree(degree_array: list, novelty_array: list, step: float, prev_degree=None) -> tuple:
    """Returns (optimal_degree, uncertainty).

    Smooths novelty_array with a triangular [0.25, 0.5, 0.25] kernel
    (edge-clamped boundaries), then runs the unmodified grouping/argmin
    algorithm from find_optimal_degree on the smoothed values.

    `prev_degree` is unused by this strategy (accepted for signature
    compatibility with the shared evaluation harness).
    """
    smoothed = _smooth_triangular(novelty_array)
    return _grouping_argmin(degree_array, smoothed, step)
