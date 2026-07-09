"""
Strategy 12: Sliding-window region selection + parabolic sub-bin interpolation.

Combines strategy 5 (robustness: pick the contiguous window of W samples with
the lowest total novelty, so a single spurious low sample surrounded by high
neighbors can't hijack the estimate) with strategy 3 (precision: fit a
parabola through the true minimum sample and its immediate neighbors for a
continuous-resolution offset within the scan step).

RESULT: this combination does NOT beat strategy 3 alone on the reference
trajectory. Three variants were tried, all worse than plain parabolic
interpolation (MAE 6.9653):
  v1 -- parabola centered on the window's geometric center: MAE 7.3774
  v2 (kept here) -- window picks a candidate region, parabola still centers
      on the true local minimum inside it: MAE 7.6285
  v3 -- window-sum used only as the tie-break among exact-minimum ties
      (replacing "closest to 0 degrees"): MAE 8.1373, the worst of the three

Root cause: strategy 3's "closest to 0 degrees" tie-break is exercised on
158/200 frames (exact-value ties at the minimum are the norm, not the
exception, given the small integer novelty range) and is doing real,
load-bearing work -- it encodes an effective implicit prior ("when
ambiguous, prefer continuing straight ahead"), which fits this corridor-like
trajectory well. Every attempt to override or dilute that prior with a
window/robustness criterion made things worse, not better: the two
strategies' individual gains (3's sub-bin precision, 5's noise robustness)
do not compose additively when layered on the same decision point.

Standalone module: only the Python standard library is used (no numpy
required), so it can be loaded directly via
importlib.util.spec_from_file_location without importing the insect_nav
package.
"""

W = 3  # window width in samples; must be odd for a well-defined center


def select_degree(degree_array: list, novelty_array: list, step: float, prev_degree=None) -> tuple:
    """Returns (optimal_degree, uncertainty)."""
    n = len(novelty_array)
    w = min(W, n)
    if w % 2 == 0:
        w -= 1
    w = max(w, 1)

    window_sums = [sum(novelty_array[s:s + w]) for s in range(n - w + 1)]
    best_sum = min(window_sums)
    best_starts = [s for s, v in enumerate(window_sums) if v == best_sum]

    def window_center_degree(s):
        idxs = range(s, s + w)
        return sum(degree_array[i] for i in idxs) / w

    best_start = min(best_starts, key=lambda s: abs(window_center_degree(s)))

    # Within the winning (robust) window, anchor on the actual lowest-novelty
    # sample rather than the window's arithmetic center.
    window_idxs = range(best_start, best_start + w)
    local_min = min(novelty_array[i] for i in window_idxs)
    idx = min(
        (i for i in window_idxs if novelty_array[i] == local_min),
        key=lambda i: abs(degree_array[i]),
    )

    y0 = novelty_array[idx]
    y_m1 = novelty_array[idx - 1] if idx - 1 >= 0 else y0
    y_p1 = novelty_array[idx + 1] if idx + 1 < n else y0

    denom = (y_m1 - 2 * y0 + y_p1)
    delta = 0.5 * (y_m1 - y_p1) / denom if denom != 0 else 0.0
    delta = max(-0.5, min(0.5, delta))

    optimal_degree = degree_array[idx] + delta * step

    max_n = max(novelty_array)
    uncertainty = best_sum / (w * max_n) if max_n > 0 else 0.0

    return optimal_degree, uncertainty
