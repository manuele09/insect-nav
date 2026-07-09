"""
Strategy 3: Parabolic sub-bin interpolation.

Standard sub-sample valley refinement (as used e.g. in cross-correlation peak
fitting): finds the global-minimum-novelty shift (ties broken by proximity to
0 degrees, same as the production `find_optimal_degree`), then fits a
parabola through that sample and its two immediate neighbors to estimate a
continuous-resolution offset within the scan step, instead of being locked to
multiples of the 9-degree scan step.

Standalone module: only the Python standard library and numpy are used, so
it can be loaded directly via importlib.util.spec_from_file_location without
importing the (partially broken) insect_nav package.
"""


def select_degree(degree_array: list, novelty_array: list, step: float, prev_degree=None) -> tuple:
    """Returns (optimal_degree, uncertainty)."""
    min_value = min(novelty_array)
    tied_idx = [i for i, v in enumerate(novelty_array) if v == min_value]
    idx = min(tied_idx, key=lambda i: abs(degree_array[i]))  # among ties, closest to 0 degrees

    y0 = novelty_array[idx]
    y_m1 = novelty_array[idx - 1] if idx - 1 >= 0 else y0  # clamp at array edges
    y_p1 = novelty_array[idx + 1] if idx + 1 < len(novelty_array) else y0

    denom = (y_m1 - 2 * y0 + y_p1)
    delta = 0.5 * (y_m1 - y_p1) / denom if denom != 0 else 0.0
    delta = max(-0.5, min(0.5, delta))  # clip to stay within the adjacent half-bin

    optimal_degree = degree_array[idx] + delta * step

    uncertainty = 1 - (1 / len(tied_idx))  # fraction of tie competitors beyond the chosen one; 0 if unique minimum

    return optimal_degree, uncertainty
