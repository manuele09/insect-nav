"""
Strategy 13: Level-and-density weighted centroid.

Unlike strategy 1 (softmax) / strategy 2 (population vector) / strategy 4
(margin average) -- which weight each *individual sample* by its novelty
value and were all substantially worse than baseline on both reference
trajectories, because a lone noisy sample at a "good" level gets the same
say as a large coherent cluster -- this strategy weights whole *novelty
levels* by two independent signals:

  1. how good the level is (how close to the minimum novelty value), and
  2. how many points support that level (its total density across the scan),

and, within each level, still finds its most spatially-coherent cluster
(the largest run of step-adjacent degrees at that level, same consecutive-
run logic as the baseline's `find_optimal_degree`) to anchor that level's
representative heading -- so an isolated stray point at a decent novelty
level does not by itself pull the estimate, but a *wide* low-novelty
plateau (many points, even if not the strict global minimum) can.

Example this is meant to capture: min novelty 0 hit by a single isolated
shift, while novelty 1 is hit by a wide 5-shift consecutive plateau --
baseline commits entirely to the isolated single-shift novelty-0 point;
this strategy lets the much larger and nearly-as-good novelty-1 plateau
pull the estimate toward it.

Standalone module: only the Python standard library and numpy are used, so
it can be loaded directly via importlib.util.spec_from_file_location without
importing the (partially broken) insect_nav package.
"""
import math

T = 0.2  # decay temperature for level "goodness" (exp(-(level-min)/T)); smaller = faster falloff, i.e. closer to baseline-only-min behavior.
# Swept 0.05-1.0 on both reference trajectories (Go2, Pioneer): T=0.2 sits at/near
# the MAE minimum on both (Go2 7.2566 vs baseline 7.2773; Pioneer 17.7697 vs
# baseline 18.0759) -- small but consistent, unlike the point-wise-weighted
# strategies (1, 2, 4, 8, 10) which all got substantially worse on both.


def _largest_consecutive_group(points_degrees, step):
    groups, current = [], [points_degrees[0]]
    for d in points_degrees[1:]:
        if abs(d - current[-1]) == step:
            current.append(d)
        else:
            groups.append(current)
            current = [d]
    groups.append(current)

    max_len = max(len(g) for g in groups)
    longest = [g for g in groups if len(g) == max_len]
    best_group = longest[0] if len(longest) == 1 else min(longest, key=lambda g: abs(sum(g) / len(g)))
    return best_group


def select_degree(degree_array: list, novelty_array: list, step: float, prev_degree=None) -> tuple:
    """Returns (optimal_degree, uncertainty)."""
    min_value = min(novelty_array)
    levels = sorted(set(novelty_array))

    level_centroids = []
    level_weights = []
    level_sizes = []

    for level in levels:
        idxs = [i for i, v in enumerate(novelty_array) if v == level]
        pts_degrees = [degree_array[i] for i in idxs]

        best_group = _largest_consecutive_group(pts_degrees, step)
        centroid = sum(best_group) / len(best_group)

        size = len(idxs)  # total point count at this level (density), not just the winning cluster
        goodness = math.exp(-(level - min_value) / T)
        weight = size * goodness

        level_centroids.append(centroid)
        level_weights.append(weight)
        level_sizes.append(size)

    total_weight = sum(level_weights)
    optimal_degree = sum(w * c for w, c in zip(level_weights, level_centroids)) / total_weight

    min_level_weight = level_weights[0]  # levels sorted ascending -> index 0 is min_value's level
    uncertainty = 1 - (min_level_weight / total_weight)

    return optimal_degree, uncertainty
