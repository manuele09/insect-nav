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

import importlib
import os
import re


def available_strategies() -> list:
    """Sorted list of strategy module basenames in this package (e.g.
    ['strategy1_softmax_centroid', ...])."""
    pkg_dir = os.path.dirname(__file__)
    names = [
        fn[:-3] for fn in os.listdir(pkg_dir)
        if fn.endswith(".py") and not fn.startswith("_")
    ]
    return sorted(names, key=lambda s: (len(s.split("_")[0]), s))


def load_strategy(name: str):
    """Resolve a degree-selection strategy by name and return its
    ``select_degree(degree_array, novelty_array, step, prev_degree=None)``
    callable (the shared contract of every module in this package).

    ``name`` may be the full module basename ("strategy3_parabolic_interp"),
    the leading token ("strategy3"), or just the number ("3"). Matching is
    case-insensitive. Raises ValueError on an unknown/ambiguous name.

    This is the resolver used by NeuralModelBase to honour the
    INSECT_NAV_DEGREE_STRATEGY environment variable (opt-in; unset keeps the
    built-in find_optimal_degree grouping logic).
    """
    token = str(name).strip().lower()
    if not token:
        raise ValueError("empty degree-strategy name")
    if re.fullmatch(r"\d+", token):
        token = "strategy" + token

    matches = []
    for base in available_strategies():
        # base == token              -> full basename or bare "strategyN"
        # base.startswith(token+"_") -> "strategyN" matching "strategyN_desc"
        #   (the trailing "_" keeps "strategy1" from matching "strategy10_...")
        if base == token or base.startswith(token + "_"):
            matches.append(base)

    if not matches:
        raise ValueError(
            f"unknown degree strategy {name!r}; available: {available_strategies()}"
        )
    if len(matches) > 1:
        raise ValueError(f"ambiguous degree strategy {name!r}; matches: {matches}")

    module = importlib.import_module(f"{__name__}.{matches[0]}")
    return module.select_degree
