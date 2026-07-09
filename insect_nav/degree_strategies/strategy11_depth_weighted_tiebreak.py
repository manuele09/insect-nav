"""
Strategy 11: identical to the baseline `find_optimal_degree` algorithm, with
ONE change to the tie-break rule used when multiple longest groups of tied
(minimally-novel) headings exist.

Baseline tie-break: pick the group whose mean is closest to 0 degrees --
this ignores how "distinct"/deep each candidate minimum actually is.

This strategy's tie-break: pick the group with the greatest LOCAL DEPTH,
i.e. how much lower its novelty plateau is compared to the novelty values
immediately flanking it on either side (the closer of the two neighbors).
A deep, sharply-bounded minimum (surrounded by much higher novelty) is a
more distinct/trustworthy signal than a shallow one that blends into its
neighbors, even if both groups tie in raw length. If depth is still tied
among multiple groups, fall back to the original "closest to 0 degrees"
rule as a final tie-break.

`prev_degree` is accepted for signature compatibility but is unused by
this strategy.

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
    else:
        # TIE-BREAK RULE: greatest local depth (how much lower the group's
        # novelty plateau is compared to its flanking neighbors).
        def local_depth(g):
            first_idx = degree_array.index(g[0])
            last_idx = degree_array.index(g[-1])
            left_neighbor_novelty = (
                novelty_array[first_idx - 1] if first_idx > 0 else float('inf')
            )
            right_neighbor_novelty = (
                novelty_array[last_idx + 1] if last_idx < len(novelty_array) - 1 else float('inf')
            )
            return min(left_neighbor_novelty, right_neighbor_novelty) - min_value

        depths = [local_depth(g) for g in longest]
        max_depth = max(depths)
        deepest = [g for g, d in zip(longest, depths) if d == max_depth]

        if len(deepest) == 1:
            best_group = deepest[0]
        else:
            # Final fallback: closest to 0 degrees
            best_group = min(deepest, key=lambda g: abs(np.mean(g)))

    optimal_degree = float(np.mean(best_group))
    indecision_a = 1 - len(best_group) / len(deg_min)
    indecision_b = (len(best_group) - 1) / len(degree_array)
    uncertainty = (indecision_a + indecision_b) / 2
    return optimal_degree, uncertainty
