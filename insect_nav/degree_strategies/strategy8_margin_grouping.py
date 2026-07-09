"""
Strategy 8: Margin-tolerant thresholding + baseline grouping.

Keeps the ENTIRE baseline algorithm (find_optimal_degree in insect_nav/base.py)
structure identical -- consecutive-run grouping of candidate degrees, longest-
group selection, closest-to-zero tie-break, and the same uncertainty formula.
The ONLY change is the initial thresholding step: instead of requiring exact
equality to the scan minimum (`v == min_value`), any shift within `MARGIN`
spike counts of the minimum (`v <= min_value + MARGIN`) is admitted into the
candidate set that then feeds the (unchanged) consecutive-run grouping logic.

degree_array / novelty_array are assumed to already be in ascending-degree
order, so the broadened candidate set (deg_min) stays naturally sorted and
the adjacency check `abs(d - current[-1]) == step` works unchanged.

prev_degree is accepted for signature compatibility with the shared
evaluation harness but is not used by this strategy (it is stateless).

Standalone module: only stdlib + numpy, no insect_nav imports, so it can be
loaded directly via importlib without pulling in insect_nav's dependency
chain.
"""
from statistics import mean

MARGIN = 1  # tolerance in spike-count units (hyperparameter)


def select_degree(degree_array: list, novelty_array: list, step: float, prev_degree=None) -> tuple:
    """Returns (optimal_degree, uncertainty)."""
    min_value = min(novelty_array)
    deg_min = [d for d, v in zip(degree_array, novelty_array) if v <= min_value + MARGIN]

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
    uncertainty = (indecision_a + indecision_b) / 2
    return optimal_degree, uncertainty
