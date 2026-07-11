"""Distribuzione del tempo assoluto (da t=0) al 2° spike MBON, sulla traiettoria di test."""

import csv
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common_paths import TRAJECTORY_MBON_VOLTAGE_PROFILE_DIR  # noqa: E402
from insect_nav.plot_style import (                            # noqa: E402
    POPULATION_COLORS,
    apply_style,
    new_figure,
    save_figure,
    style_axes,
)

OUTPUT_DIR = TRAJECTORY_MBON_VOLTAGE_PROFILE_DIR
CSV_PATH = os.path.join(OUTPUT_DIR, "mbon_voltage_trials.csv")


def main():
    apply_style()

    rows = list(csv.DictReader(open(CSV_PATH)))
    counts = np.array([int(r["spike_count"]) for r in rows])
    n_zero = int(np.sum(counts == 0))
    n_one = int(np.sum(counts == 1))
    values = np.array([
        float(r["t_second_spike_from_sim_start_ms"])
        for r in rows if r["t_second_spike_from_sim_start_ms"] not in ("", "nan")
    ])
    n_total = len(rows)
    print(f"Prove con 2o spike: {len(values)}/{n_total}")
    print(f"Prove a 0 spike: {n_zero}, a esattamente 1 spike: {n_one}")
    print(f"Media: {values.mean():.2f} ms, std: {values.std():.2f} ms, "
          f"min: {values.min():.2f} ms, max: {values.max():.2f} ms")

    fig, ax = new_figure("error_vs_x")
    ax.hist(values, bins=20, color=POPULATION_COLORS["MBON"], edgecolor="black", linewidth=0.6, alpha=0.85)
    ax.axvline(values.mean(), color="black", linestyle="--", linewidth=1.5,
               label=f"Media = {values.mean():.2f} ms")
    ax.plot([], [], " ", label=f"Mai arrivate a 1 spike (0 spike): {n_zero}/{n_total}")
    ax.plot([], [], " ", label=f"Mai arrivate a 2 spike (esattamente 1 spike): {n_one}/{n_total}")
    style_axes(ax, xlabel="Tempo al 2° spike da t=0 [ms]", ylabel="Numero di prove",
               title=f"Distribuzione tempo al 2° spike (n={len(values)}/{n_total})")
    ax.legend(loc="best", frameon=True, framealpha=0.9, edgecolor="0.3")
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "metriche_2_spike", "second_spike_time_distribution.png")
    save_figure(fig, out_path)
    print(f"Figura salvata in: {out_path}")


if __name__ == "__main__":
    main()
