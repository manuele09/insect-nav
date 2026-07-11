"""
Trova la prova con spike rate piu' alta E pendenza di picco (peak_slope,
onset->picco) piu' bassa (esclusi valori nulli/NaN) -- il punto che piu'
"tradisce" la relazione generale (di solito positiva) tra pendenza e rate.

Criterio: rango congiunto = rango(rate, decrescente) + rango(|peak_slope|,
crescente), minimizzato -- soddisfa entrambi i criteri insieme senza far
dominare uno dei due per differenze di scala.

Rilancia quella specifica prova (stesso frame_id/shift_deg, stesso
PRESENT_TIME_MS=40 in memoria, stesso setup CPU deterministico) per
recuperare la traccia di tensione MBON completa e la plotta.

Richiede pygenn -- va eseguito dentro il container distrobox:
    distrobox enter insect-navContainer -- python tests/mbon_voltage_profile/scripts/custom_high_rate_low_slope.py
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
    TRAJECTORY_MBON_VOLTAGE_PROFILE_DIR,
    TRAJECTORY_PANORAMA_DIR,
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

OUTPUT_DIR = TRAJECTORY_MBON_VOLTAGE_PROFILE_DIR
CSV_PATH = os.path.join(OUTPUT_DIR, "mbon_voltage_trials.csv")
NEW_PRESENT_TIME_MS = 40.0


def load_params_without_touching_disk(path):
    with open(path) as f:
        parameters = json.load(f)
    return update_paths(parameters, path)


def main():
    apply_style()

    rows = list(csv.DictReader(open(CSV_PATH)))

    def col(name):
        return np.array([float(r[name]) if r[name] not in ("", "nan") else np.nan for r in rows])

    mean_isi = col("mean_isi_ms")
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(mean_isi > 0, 1000.0 / mean_isi, np.nan)
    peak_slope = col("peak_slope_mV_per_ms")

    valid = ~np.isnan(rate) & ~np.isnan(peak_slope) & (peak_slope != 0)
    print(f"Prove valide (rate definita, pendenza != 0): {valid.sum()}/{len(rows)}")

    idx_valid = np.nonzero(valid)[0]
    rate_v = rate[idx_valid]
    slope_v = np.abs(peak_slope[idx_valid])

    rate_rank = np.argsort(np.argsort(-rate_v))       # 0 = rate piu' alta
    slope_rank = np.argsort(np.argsort(slope_v))       # 0 = pendenza piu' bassa
    combined = rate_rank + slope_rank
    best_local = int(np.argmin(combined))
    best_idx = idx_valid[best_local]

    row = rows[best_idx]
    frame_id = int(row["frame_id"])
    shift_deg = float(row["shift_deg"])
    found_rate = rate[best_idx]
    found_slope = peak_slope[best_idx]
    found_count = int(row["spike_count"])
    print(f"Trovato: frame_id={frame_id}, shift={shift_deg:.0f}°, "
          f"rate={found_rate:.1f} Hz, peak_slope={found_slope:.2f} mV/ms, "
          f"spike_count={found_count} "
          f"(rank rate={rate_rank[best_local]}, rank slope={slope_rank[best_local]})")

    # ── rilancio la sequenza storica completa fino alla prova target ───────
    # NOTA: nn.test() non e' riproducibile in isolamento -- rilanciare la
    # singola prova (frame_id, shift_deg) su una rete appena creata da'
    # risultati leggermente diversi (es. spike_count=13 invece di 14 per
    # questo trial) rispetto a quanto registrato nel CSV originale, perche'
    # esiste uno stato che si accumula attraverso le chiamate a test()
    # successive (confermato non essere il peso kc_mbon, che resta
    # bit-identico). Per riottenere ESATTAMENTE la traccia della prova
    # registrata nel CSV, replichiamo l'intera sequenza di chiamate nello
    # stesso ordine dello script di generazione dati (tutti i frame da 0 al
    # target, con tutti gli shift, nello stesso ordine). Vedi anche
    # tests/test_mbon_reset_determinism.py.
    #
    # NOTA 2 (in corso di indagine, non ancora risolta): oltre a questo, e'
    # stato osservato che il replay completo NON e' nemmeno riproducibile
    # run-to-run (stessa sequenza, stesso codice, processi separati possono
    # dare risultati diversi) -- quindi anche con il replay qui sotto non e'
    # garantito che spike_count torni a combaciare col CSV con cui e' stato
    # rigenerato piu' di recente.
    params = load_params_without_touching_disk(PARAMS_PATH)
    params["PRESENT_TIME_MS"] = NEW_PRESENT_TIME_MS
    nn = NeuralNetwork(params, load_net={"pn_kc": True, "kc_mbon": True}, use_gpu=False)
    nn.logger.update_config("voltages", {"mbon": True})

    vrest = params["LIF_PARAMS"]["Vrest"]
    vthresh = params["LIF_PARAMS"]["Vthresh"]

    shift_degrees = list(np.arange(-180, 180, 9))
    spike_count = None
    for fid in range(0, frame_id + 1):
        frame = loadFrame(fid, frames_dir=TRAJECTORY_PANORAMA_DIR)
        for shift in shift_degrees:
            spike_count = nn.test(frame, shift_degree=shift)
            if fid == frame_id and shift == shift_deg:
                break
        if fid == frame_id:
            break

    assert spike_count == found_count, (
        f"riproduzione non deterministica? spike_count={spike_count} atteso {found_count}"
    )

    v_data = nn.logger.get_voltages("mbon")
    s_data = nn.logger.get_spikes("mbon")
    voltage = v_data["voltages"][0]
    time_axis = v_data["time_axis"]
    spike_times = s_data["times"]

    fig, ax = new_figure("error_vs_x")
    ax.plot(time_axis, voltage, color=POPULATION_COLORS["MBON"], linewidth=1.8)
    for st in spike_times:
        ax.axvline(st, color=COLORS["actual"], linestyle=":", linewidth=1.0, alpha=0.7)
    ax.axhline(vrest, color=COLORS["mean_reference"], linestyle=":", linewidth=1.0)
    ax.axhline(vthresh, color=COLORS["mean_reference"], linestyle="--", linewidth=1.0)
    ax.text(time_axis[-1], vrest, " Vrest", va="center", ha="left", fontsize=8, color=COLORS["mean_reference"])
    ax.text(time_axis[-1], vthresh, " Vthresh", va="center", ha="left", fontsize=8, color=COLORS["mean_reference"])
    style_axes(ax, xlabel="Tempo [ms]", ylabel="Tensione MBON [mV]",
               title=(f"frame={frame_id}, shift={shift_deg:.0f}° — rate={found_rate:.0f} Hz, "
                      f"pendenza={found_slope:.2f} mV/ms, {found_count} spike"))
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "metriche_2_spike", "custom_high_rate_low_slope.png")
    save_figure(fig, out_path)
    print(f"Figura salvata in: {out_path}")

    nn.model.unload()


if __name__ == "__main__":
    main()
