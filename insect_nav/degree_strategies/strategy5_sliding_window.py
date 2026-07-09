"""
Strategy 5: Sliding-window minimum-sum.

Instead of relying on the global minimum novelty value (which can be hit by
a single noisy sample), this strategy scans all contiguous windows of width
W across the novelty array and picks the window with the lowest total
novelty. The optimal degree is then the novelty-weighted centroid of that
window. This makes the estimate more robust to a single isolated low
outlier surrounded by high-novelty neighbors.

Standalone module: only the Python standard library and numpy are used, so
it can be loaded directly via importlib without importing the insect_nav
package (whose dependency chain may be broken in some environments).
"""

W = 3  # window width in samples (= 27 degrees at the default 9-degree step)


def select_degree(degree_array: list, novelty_array: list, step: float, prev_degree=None) -> tuple:
    """Returns (optimal_degree, uncertainty)."""
    n = len(novelty_array)
    w = min(W, n)

    window_sums = [sum(novelty_array[s:s + w]) for s in range(n - w + 1)]
    best_sum = min(window_sums)
    best_starts = [s for s, v in enumerate(window_sums) if v == best_sum]

    def window_center_degree(s):
        idxs = range(s, s + w)
        return sum(degree_array[i] for i in idxs) / w

    best_start = min(best_starts, key=lambda s: abs(window_center_degree(s)))

    window_idxs = list(range(best_start, best_start + w))
    max_n = max(novelty_array)
    weights = [max_n - novelty_array[i] for i in window_idxs]
    if sum(weights) == 0:
        weights = [1.0] * len(window_idxs)
    optimal_degree = sum(wt * degree_array[i] for wt, i in zip(weights, window_idxs)) / sum(weights)

    uncertainty = best_sum / (w * max_n) if max_n > 0 else 0.0

    return optimal_degree, uncertainty
