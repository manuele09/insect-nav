"""
Analisi dei casi a 0 spike MBON, testando la rete al suo PRESENT_TIME_MS
nativo (20 ms, quello di parameters.json -- non alterato nemmeno nel dict).

200 frame x 40 shift (360°) della traiettoria di test; isoliamo solo le
prove con spike_count == 0 e ne visualizziamo il profilo di tensione MBON.

Legge/scrive sotto insect-nav/tests/mbon_voltage_profile/ (vedi
common_paths.py).

Richiede pygenn -- va eseguito dentro il container distrobox:
    distrobox enter insect-navContainer -- python tests/mbon_voltage_profile/scripts/zero_spike_20ms.py
"""

import csv
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common_paths import (                                # noqa: E402
    PARAMS_PATH,
    TRAJECTORY_PANORAMA_DIR,
    TRAJECTORY_ZERO_SPIKE_20MS_DIR,
)
from insect_nav.parameters import update_paths          # noqa: E402
from insect_nav.plot_style import (                      # noqa: E402
    COLORS,
    POPULATION_COLORS,
    apply_style,
    new_figure,
    save_figure,
    style_axes,
)
from insect_nav.spiking import NeuralNetwork              # noqa: E402
from insect_nav.vision import loadFrame                   # noqa: E402

OUTPUT_DIR = TRAJECTORY_ZERO_SPIKE_20MS_DIR


def load_params_without_touching_disk(path):
    with open(path) as f:
        parameters = json.load(f)
    return update_paths(parameters, path)


def main():
    apply_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mtime_before = os.path.getmtime(PARAMS_PATH)
    params = load_params_without_touching_disk(PARAMS_PATH)
    present_time_ms = params["PRESENT_TIME_MS"]  # nativo, 20 ms -- NON alterato
    print(f"PRESENT_TIME_MS usato per questa run (nativo, non alterato): {present_time_ms} ms")

    nn = NeuralNetwork(params, load_net={"pn_kc": True, "kc_mbon": True}, use_gpu=False)
    nn.logger.update_config("voltages", {"mbon": True})

    vrest = params["LIF_PARAMS"]["Vrest"]
    vthresh = params["LIF_PARAMS"]["Vthresh"]

    frame_ids = list(range(0, 200))
    shift_degrees = list(np.arange(-180, 180, 9))
    total = len(frame_ids) * len(shift_degrees)
    print(f"Totale prove: {total}")

    zero_spike_trials = []
    n_nonzero = 0

    for frame_id in frame_ids:
        frame = loadFrame(frame_id, frames_dir=TRAJECTORY_PANORAMA_DIR)
        for shift in shift_degrees:
            spike_count = nn.test(frame, shift_degree=shift)
            if spike_count == 0:
                v_data = nn.logger.get_voltages("mbon")
                voltage = v_data["voltages"][0].copy()
                time_axis = v_data["time_axis"].copy()
                max_v = float(voltage.max())
                zero_spike_trials.append((frame_id, shift, time_axis, voltage, max_v))
            else:
                n_nonzero += 1

    n_zero = len(zero_spike_trials)
    print(f"Prove a 0 spike: {n_zero}/{total} (a {present_time_ms:.0f} ms)")

    mtime_after = os.path.getmtime(PARAMS_PATH)
    assert mtime_before == mtime_after, "parameters.json e' stato toccato! (non doveva succedere)"
    print("Verificato: parameters.json non e' stato modificato.")

    csv_path = os.path.join(OUTPUT_DIR, "zero_spike_trials.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "shift_deg", "max_voltage_mV", "distance_from_thresh_mV"])
        for frame_id, shift, _, _, max_v in zero_spike_trials:
            writer.writerow([frame_id, shift, max_v, vthresh - max_v])
    print(f"CSV salvato in: {csv_path}")

    if n_zero == 0:
        print("Nessuna prova a 0 spike a questo PRESENT_TIME_MS.")
        nn.model.unload()
        return

    fig, ax = new_figure("error_vs_x")
    for frame_id, shift, time_axis, voltage, max_v in zero_spike_trials:
        ax.plot(time_axis, voltage, color=POPULATION_COLORS["MBON"], alpha=0.3, linewidth=1.0)
    ax.axhline(vrest, color=COLORS["mean_reference"], linestyle=":", linewidth=1.0)
    ax.axhline(vthresh, color=COLORS["mean_reference"], linestyle="--", linewidth=1.0)
    ax.text(time_axis[-1], vrest, " Vrest", va="center", ha="left", fontsize=8, color=COLORS["mean_reference"])
    ax.text(time_axis[-1], vthresh, " Vthresh", va="center", ha="left", fontsize=8, color=COLORS["mean_reference"])
    style_axes(ax, xlabel="Tempo [ms]", ylabel="Tensione MBON [mV]",
               title=f"Tracce MBON a 0 spike, {present_time_ms:.0f} ms (n={n_zero}/{total})")
    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, "zero_spike_overlay.png"))

    max_vs = np.array([m for *_, m in zero_spike_trials])
    dist = vthresh - max_vs
    fig, ax = new_figure("error_vs_x")
    ax.hist(dist, bins=min(30, n_zero), color=POPULATION_COLORS["MBON"], edgecolor="black",
            linewidth=0.4, alpha=0.85)
    style_axes(ax, xlabel="Distanza dalla soglia al picco massimo [mV]", ylabel="Numero di prove",
               title=f"Quanto vicino alla soglia sono arrivate le prove a 0 spike (n={n_zero})")
    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, "zero_spike_distance_from_threshold.png"))

    picks = sorted(zero_spike_trials, key=lambda x: x[4])
    n_picks = min(4, n_zero)
    idxs = np.linspace(0, n_zero - 1, n_picks).round().astype(int)
    fig, axes = new_figure("multi_vertical", nrows=n_picks)
    if n_picks == 1:
        axes = [axes]
    for ax, i in zip(axes, idxs):
        frame_id, shift, time_axis, voltage, max_v = picks[i]
        ax.plot(time_axis, voltage, color=POPULATION_COLORS["MBON"], linewidth=1.6)
        ax.axhline(vrest, color=COLORS["mean_reference"], linestyle=":", linewidth=1.0)
        ax.axhline(vthresh, color=COLORS["mean_reference"], linestyle="--", linewidth=1.0)
        style_axes(ax, ylabel="V [mV]",
                   title=f"frame={frame_id}, shift={shift:.0f}°, picco={max_v:.2f} mV "
                         f"(a {vthresh - max_v:.2f} mV dalla soglia)")
    axes[-1].set_xlabel("Tempo [ms]")
    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, "zero_spike_examples.png"))

    print(f"Grafici salvati in: {OUTPUT_DIR}")
    nn.model.unload()


if __name__ == "__main__":
    main()
