"""
Strategy 9: Confidence-gated temporal EMA.

Runs the baseline `find_optimal_degree` algorithm (copied verbatim as
`_baseline_select`) to get a raw per-frame heading estimate and its
uncertainty, then blends that raw estimate with the previous frame's final
chosen heading. The blend weight (alpha) is derived from the baseline's own
uncertainty signal: confident frames (low uncertainty, narrow/well-defined
minimum) trust the raw estimate more; ambiguous frames (high uncertainty,
wide/flat plateau) lean more on the previous frame's heading.

Standalone module: only stdlib + numpy, no insect_nav imports, so it can be
loaded directly via importlib.util.spec_from_file_location.
"""
from statistics import mean


def _baseline_select(degree_array, novelty_array, step):
    """Verbatim copy of insect_nav/base.py::find_optimal_degree."""
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
    best_group = longest[0] if len(longest) == 1 else min(longest, key=lambda g: abs(mean(g)))

    optimal_degree = mean(best_group)
    indecision_a = 1 - len(best_group) / len(deg_min)
    indecision_b = (len(best_group) - 1) / len(degree_array)
    uncertainty = (indecision_a + indecision_b) / 2   # higher = more ambiguous/uncertain decision this frame
    return optimal_degree, uncertainty


def select_degree(degree_array: list, novelty_array: list, step: float, prev_degree=None) -> tuple:
    """Returns (optimal_degree, uncertainty)."""
    raw_degree, raw_uncertainty = _baseline_select(degree_array, novelty_array, step)

    if prev_degree is None:
        optimal_degree = raw_degree
    else:
        alpha = 1 - raw_uncertainty
        alpha = max(0.2, min(1.0, alpha))   # clip: never fully ignore the new measurement, never fully ignore history
        optimal_degree = alpha * raw_degree + (1 - alpha) * prev_degree

    uncertainty = raw_uncertainty
    return optimal_degree, uncertainty
