"""Alternative strategies for selecting the optimal heading from a novelty_array.

Each sibling module exposes a single function:

    def select_degree(degree_array: list[float],
                       novelty_array: list[float],
                       step: float,
                       prev_degree: float | None = None) -> tuple[float, float]:
        ...
        return optimal_degree, uncertainty

matching the (degree_array, novelty_array) contract of
insect_nav.base.NeuralModelBase.find_optimal_degree, plus an optional
prev_degree for strategies that use temporal context across frames.

Modules are intentionally dependency-free (stdlib + numpy only, no
`insect_nav` imports) so they can be benchmarked on the host Python without
the pygenn/distrobox environment -- see
insect_nav/tuning/degree_strategy_eval.py for the shared evaluation harness.
"""
