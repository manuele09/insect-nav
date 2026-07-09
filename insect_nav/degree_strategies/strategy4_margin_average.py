"""
Strategy 4: Margin-based flat average.

Softens the baseline's hard "exact minimum novelty only" threshold into a
small tolerance margin: any shift whose novelty is within `MARGIN` spike
counts of the scan's minimum is considered "good enough" and included in a
plain, unweighted arithmetic mean of degrees. Unlike the baseline
(find_optimal_degree in insect_nav/base.py), there is no adjacency /
consecutive-run grouping step at all -- every qualifying shift counts,
regardless of whether it is contiguous with the others.

uncertainty is the fraction of the scan considered "good enough" (higher =
more shifts qualify = more spread out / less decisive).

prev_degree is accepted for signature compatibility with the shared
evaluation harness but is not used by this strategy (it is stateless).

Standalone module: only stdlib + numpy, no insect_nav imports, so it can be
loaded directly via importlib without pulling in insect_nav's dependency
chain.
"""
import numpy as np

MARGIN = 1  # tolerance in spike-count units (hyperparameter)


def select_degree(degree_array: list, novelty_array: list, step: float, prev_degree=None) -> tuple:
    """Returns (optimal_degree, uncertainty)."""
    novelty = np.asarray(novelty_array, dtype=np.float64)
    degrees_arr = np.asarray(degree_array, dtype=np.float64)

    threshold = np.min(novelty) + MARGIN
    mask = novelty <= threshold

    selected = degrees_arr[mask]
    optimal_degree = float(np.mean(selected))
    uncertainty = float(len(selected) / len(degree_array))

    return optimal_degree, uncertainty
