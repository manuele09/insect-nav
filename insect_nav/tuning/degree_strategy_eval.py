"""
Evaluation harness for alternative `find_optimal_degree` strategies.

Reuses the novelty arrays already logged during a live simulation run
(test_log.csv) instead of re-running the spiking network, so any strategy
implementing the standard signature below can be benchmarked in pure Python,
no pygenn required:

    def select_degree(degree_array: list[float],
                       novelty_array: list[float],
                       step: float,
                       prev_degree: float | None = None) -> tuple[float, float]:
        ...
        return optimal_degree, uncertainty

`prev_degree` is the previous frame's chosen optimal_degree (None for the
first frame) -- ignored by strategies that don't use temporal context.

For each frame:
    desired_yaw = pose_yaw + radians(-optimal_degree)
    error = circular_diff(desired_yaw, optimal_yaw)   # degrees, absolute

matching the convention used by control_navigator.py to turn a network
decision into an absolute heading and by nav_vectors.csv's optimal_yaw
(ground truth heading from the training-trajectory spline).
"""
import csv
import math
import os

import numpy as np

DEFAULT_TRAJ_DIR = (
    "/home/emanuele/Desktop/Test Polo Sim/Go2/"
    "t95_f7_w0.1698610745174438_v40.426075822710246_ver0.5_hor0.09624930067328646_ts2_seed7/"
    "plots/simulation/traj_00_10-58-47"
)


def _circular_diff_deg(a_rad: float, b_rad: float) -> float:
    diff = (a_rad - b_rad + math.pi) % (2 * math.pi) - math.pi
    return abs(math.degrees(diff))


def load_trajectory_data(traj_dir: str = DEFAULT_TRAJ_DIR):
    with open(os.path.join(traj_dir, "test_log.csv"), newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        test_rows = list(reader)

    with open(os.path.join(traj_dir, "pose_log.csv"), newline="") as f:
        pose_rows = list(csv.DictReader(f))

    with open(os.path.join(traj_dir, "nav_vectors.csv"), newline="") as f:
        nav_rows = list(csv.DictReader(f))

    degree_cols = [c for c in header if c.startswith("degree_")]
    degree_array = [float(c.replace("degree_", "")) for c in degree_cols]

    return {
        "degree_array": degree_array,
        "degree_cols": degree_cols,
        "test_rows": test_rows,
        "pose_rows": pose_rows,
        "nav_rows": nav_rows,
    }


def evaluate_strategy(select_degree, traj_dir: str = DEFAULT_TRAJ_DIR, step: float = 9.0,
                       stateful: bool = True) -> dict:
    """
    Run `select_degree` over every frame of the reference trajectory and
    report error statistics against optimal_yaw.

    Args:
        select_degree: callable(degree_array, novelty_array, step, prev_degree) -> (deg, uncertainty)
        stateful: if True, feeds each frame's own output as `prev_degree` to
            the next call (sequential replay, as it would run live). If
            False, prev_degree is always None (useful to test a strategy in
            isolation from frame-to-frame history).
    """
    data = load_trajectory_data(traj_dir)
    degree_array = data["degree_array"]

    errors = []
    per_frame = []
    prev_degree = None

    for i, row in enumerate(data["test_rows"]):
        novelty_array = [int(row[c]) for c in data["degree_cols"]]
        optimal_degree, uncertainty = select_degree(degree_array, novelty_array, step, prev_degree)

        pose_yaw = float(data["pose_rows"][i]["yaw"])
        desired_yaw = pose_yaw + math.radians(-optimal_degree)

        opt_str = data["nav_rows"][i]["optimal_yaw"]
        error_deg = None
        if opt_str != "":
            optimal_yaw = float(opt_str)
            error_deg = _circular_diff_deg(desired_yaw, optimal_yaw)
            errors.append(error_deg)

        per_frame.append({
            "frame": i,
            "optimal_degree": optimal_degree,
            "uncertainty": uncertainty,
            "desired_yaw": desired_yaw,
            "error_deg": error_deg,
        })

        if stateful:
            prev_degree = optimal_degree

    errors_arr = np.array(errors, dtype=np.float64)
    return {
        "n_frames": len(data["test_rows"]),
        "n_scored": len(errors),
        "mae_deg": float(np.mean(errors_arr)),
        "median_deg": float(np.median(errors_arr)),
        "std_deg": float(np.std(errors_arr)),
        "max_deg": float(np.max(errors_arr)),
        "p90_deg": float(np.percentile(errors_arr, 90)),
        "per_frame": per_frame,
    }


def baseline_find_optimal_degree(degree_array, novelty_array, step, prev_degree=None):
    """Verbatim reimplementation of insect_nav/base.py::find_optimal_degree,
    exposed with the shared (degree_array, novelty_array, step, prev_degree)
    signature so it can be run through evaluate_strategy() as the reference
    to beat. prev_degree is accepted but unused (baseline is stateless)."""
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
    best_group = longest[0] if len(longest) == 1 else min(longest, key=lambda g: abs(np.mean(g)))

    optimal_degree = float(np.mean(best_group))
    indecision_a = 1 - len(best_group) / len(deg_min)
    indecision_b = (len(best_group) - 1) / len(degree_array)
    uncertainty = (indecision_a + indecision_b) / 2

    return optimal_degree, uncertainty


def print_report(name: str, stats: dict, baseline_mae: float | None = None):
    print(f"--- {name} ---")
    print(f"  frames scored: {stats['n_scored']}/{stats['n_frames']}")
    print(f"  MAE:    {stats['mae_deg']:.4f} deg")
    print(f"  median: {stats['median_deg']:.4f} deg")
    print(f"  std:    {stats['std_deg']:.4f} deg")
    print(f"  p90:    {stats['p90_deg']:.4f} deg")
    print(f"  max:    {stats['max_deg']:.4f} deg")
    if baseline_mae is not None:
        delta = stats["mae_deg"] - baseline_mae
        pct = 100 * delta / baseline_mae
        print(f"  vs baseline MAE ({baseline_mae:.4f}): {delta:+.4f} deg ({pct:+.1f}%)")


if __name__ == "__main__":
    baseline_stats = evaluate_strategy(baseline_find_optimal_degree)
    print_report("baseline (find_optimal_degree)", baseline_stats)
