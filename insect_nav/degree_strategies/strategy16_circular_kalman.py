"""
Strategy 16: Circular Kalman filter temporal smoothing.

Formalizes the idea behind strategy 9 (confidence-gated EMA) using an actual
scalar Kalman filter instead of an ad-hoc alpha blend, matching how the
insect head-direction / ring-attractor literature models heading tracking
(a "circular Kalman filter... implements a circular Kalman filter for
statistically optimal circular estimation, with uncertainty encoded in the
amplitude of the activity bump").

Model:
    state:        theta_t (heading), with variance P_t
    process:      theta_t = theta_(t-1) + w_t,  w_t ~ N(0, Q)          (random-walk prior: heading drifts smoothly frame-to-frame)
    measurement:  z_t = theta_t + v_t,  v_t ~ N(0, R_t)                (z_t = this frame's raw baseline estimate)
    R_t = R_BASE + R_SCALE * raw_uncertainty_t                          (a low-confidence raw frame is a noisier measurement, so it is trusted less by the Kalman gain -- same intuition as strategy 9's confidence gate, but now composed correctly through the gain instead of a fixed clip)

Standard scalar Kalman update per frame:
    P_pred = P_(t-1) + Q
    K = P_pred / (P_pred + R_t)
    theta_t = theta_pred + K * circular_diff(z_t, theta_pred)
    P_t = (1 - K) * P_pred

Hyperparameters (Q, R_BASE, R_SCALE, P0) were coarse-swept on the Go2
reference trajectory and checked on Pioneer -- see
insect_nav/tuning/degree_strategy_eval.py for how to re-sweep against new
trajectories.

State caveat: the shared select_degree(...) contract only threads
`prev_degree` (not a full filter state) between calls, so the running
variance P_t is kept in a module-level dict and reset whenever
`prev_degree is None` (the harness's convention for "start of a new
trajectory replay") -- this makes the filter correct for sequential
single-trajectory replay via evaluate_strategy(), but is NOT safe for
concurrent/interleaved use across multiple trajectories in the same process
without resetting between them.

Standalone module: only the Python standard library is used, so it can be
loaded directly via importlib.util.spec_from_file_location without
importing the insect_nav package.
"""

Q = 100.0         # process noise variance (deg^2): how much the true heading can drift between frames
R_BASE = 10.0     # measurement noise variance (deg^2) at raw_uncertainty == 0
R_SCALE = 10.0    # additional measurement noise variance scaled by raw_uncertainty
P0 = 8100.0       # initial variance (deg^2), ~90 deg std -- wide/uninformative prior at trajectory start
# Grid-swept Q in [5,1000], R_BASE in [1,50], R_SCALE in [10,500] on both
# reference trajectories: EVERY combination that beat baseline on both
# converged toward high-Q/low-R (Kalman gain K -> 1, i.e. "trust the raw
# per-frame measurement almost fully"), which is nearly indistinguishable
# from no smoothing at all. Best found: Go2 7.2021 vs baseline 7.2773
# (-1.0%), Pioneer 18.0066 vs baseline 18.0759 (-0.4%) -- consistent with
# strategies 7/9/11 (all other temporal approaches tried): temporal
# smoothing does not meaningfully help on these two trajectories, no matter
# how it's formalized.

_state = {"P": None}


def _baseline_select(degree_array, novelty_array, step):
    """Verbatim copy of insect_nav/base.py::find_optimal_degree (can't import
    insect_nav.base directly, see module docstring)."""
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
    best_group = longest[0] if len(longest) == 1 else min(longest, key=lambda g: abs(sum(g) / len(g)))

    optimal_degree = sum(best_group) / len(best_group)
    indecision_a = 1 - len(best_group) / len(deg_min)
    indecision_b = (len(best_group) - 1) / len(degree_array)
    uncertainty = (indecision_a + indecision_b) / 2

    return optimal_degree, uncertainty


def _circular_diff(a, b):
    return (a - b + 180) % 360 - 180


def select_degree(degree_array: list, novelty_array: list, step: float, prev_degree=None) -> tuple:
    """Returns (optimal_degree, uncertainty)."""
    raw_degree, raw_uncertainty = _baseline_select(degree_array, novelty_array, step)

    if prev_degree is None:
        _state["P"] = P0
        return raw_degree, raw_uncertainty

    P_prev = _state["P"] if _state["P"] is not None else P0
    P_pred = P_prev + Q
    R = R_BASE + R_SCALE * raw_uncertainty

    innovation = _circular_diff(raw_degree, prev_degree)
    K = P_pred / (P_pred + R)
    optimal_degree = prev_degree + K * innovation
    optimal_degree = (optimal_degree + 180) % 360 - 180

    _state["P"] = (1 - K) * P_pred

    return optimal_degree, raw_uncertainty
