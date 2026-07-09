"""
Strategy 7: identical to the baseline `find_optimal_degree` algorithm, with
ONE change to the tie-break rule used when multiple longest groups of tied
(minimally-novel) headings exist.

Baseline tie-break: pick the group whose mean is closest to 0 degrees.
This strategy's tie-break: pick the group whose mean is closest to the
PREVIOUS frame's chosen optimal_degree (temporal continuity), falling back
to "closest to 0 degrees" when there is no previous frame (prev_degree is
None, e.g. the first frame of a trajectory).

Standard library + numpy only -- no insect_nav imports, so this module can
be loaded standalone via importlib even in environments missing some of
insect_nav's dependencies.
"""
import numpy as np


def select_degree(degree_array: list, novelty_array: list, step: float, prev_degree=None) -> tuple:
    """Returns (optimal_degree, uncertainty)."""
    min_value = min(novelty_array)
    deg_min = [d for d, v in zip(degree_array, novelty_array) if v == min_value]

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
    if len(longest) == 1:
        best_group = longest[0]
    elif prev_degree is not None:
        # TIE-BREAK RULE: closest to previous frame's chosen degree
        best_group = min(longest, key=lambda g: abs(np.mean(g) - prev_degree))
    else:
        # Fallback (no previous frame): closest to 0 degrees
        best_group = min(longest, key=lambda g: abs(np.mean(g)))

    optimal_degree = float(np.mean(best_group))
    indecision_a = 1 - len(best_group) / len(deg_min)
    indecision_b = (len(best_group) - 1) / len(degree_array)
    uncertainty = (indecision_a + indecision_b) / 2
    return optimal_degree, uncertainty
