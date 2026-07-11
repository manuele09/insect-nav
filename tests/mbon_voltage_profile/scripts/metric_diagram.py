"""
Per ciascuna delle metriche riferite al 1°, 2° e 3° spike (rise_duration,
peak_slope=rise_slope, max_slope, mean_voltage_rise -- solo 1°; + ISI tra
spike consecutivi): una figura con la definizione (traccia annotata, una
singola prova rappresentativa) affiancata dal pannello coi risultati (quella
metrica vs spike rate, sulle prove del CSV gia' generato da
mbon_voltage_profile.py). Spike rate = 1000/ISI medio, misurata dal primo
all'ultimo spike (richiede >=2 spike).

Le figure sono organizzate in sottocartelle per famiglia di metrica:
  metriche_1_spike/, metriche_2_spike/, metriche_3_spike/
I grafici generali (spike_rate vs spike_count/corrente) restano nella
cartella di output principale.

Il rapporto metrica<->rate e' modellato con un'esponenziale saturante
(y = A + B*exp(-C*x), fit non lineare) invece di una retta: i dati mostrano
chiaramente una saturazione (il rate massimo e' limitato dal periodo
refrattario del MBON), non una relazione lineare. Bonta' del fit riportata
come R² (coefficiente di determinazione).

Legge/scrive sotto insect-nav/tests/mbon_voltage_profile/ (vedi
common_paths.py).

Richiede pygenn -- va eseguito dentro il container distrobox:
    distrobox enter insect-navContainer -- python tests/mbon_voltage_profile/scripts/metric_diagram.py
"""

import csv
import json
import os
import sys

import numpy as np
from scipy.optimize import curve_fit

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common_paths import (                                     # noqa: E402
    PARAMS_PATH,
    TRAJECTORY_MBON_VOLTAGE_PROFILE_DIR,
    TRAJECTORY_PANORAMA_DIR,
)
from insect_nav.parameters import update_paths                # noqa: E402
from insect_nav.plot_style import (                            # noqa: E402
    COLORS,
    POPULATION_COLORS,
    apply_style,
    new_figure,
    save_figure,
    style_axes,
)
from insect_nav.spiking import NeuralNetwork                    # noqa: E402
from insect_nav.vision import loadFrame                         # noqa: E402


OUTPUT_DIR = TRAJECTORY_MBON_VOLTAGE_PROFILE_DIR
CSV_PATH = os.path.join(OUTPUT_DIR, "mbon_voltage_trials.csv")

DIR_1ST = os.path.join(OUTPUT_DIR, "metriche_1_spike")
DIR_2ND = os.path.join(OUTPUT_DIR, "metriche_2_spike")
DIR_3RD = os.path.join(OUTPUT_DIR, "metriche_3_spike")

NEW_PRESENT_TIME_MS = 40.0
EXAMPLE_FRAME_ID = 0
EXAMPLE_SHIFT_DEG = 90.0  # rise_duration ampia (3.7 ms) -> ben leggibile nel diagramma
RATE_YLABEL = "Spike rate [Hz]"

DATA = {}


def load_params_without_touching_disk(path):
    with open(path) as f:
        parameters = json.load(f)
    return update_paths(parameters, path)


def get_example_trace():
    params = load_params_without_touching_disk(PARAMS_PATH)
    params["PRESENT_TIME_MS"] = NEW_PRESENT_TIME_MS

    nn = NeuralNetwork(params, load_net={"pn_kc": True, "kc_mbon": True}, use_gpu=False)
    nn.logger.update_config("voltages", {"mbon": True})

    vrest = params["LIF_PARAMS"]["Vrest"]
    vthresh = params["LIF_PARAMS"]["Vthresh"]
    dt = params["DT"]

    frame = loadFrame(EXAMPLE_FRAME_ID, frames_dir=TRAJECTORY_PANORAMA_DIR)
    spike_count = nn.test(frame, shift_degree=EXAMPLE_SHIFT_DEG)

    v_data = nn.logger.get_voltages("mbon")
    s_data = nn.logger.get_spikes("mbon")
    kc_data = nn.logger.get_spikes("kc")
    voltage = v_data["voltages"][0]
    time_axis = v_data["time_axis"]
    spike_times = s_data["times"]
    t_first = float(np.min(spike_times))
    t_first_kc = float(np.min(kc_data["times"]))

    nn.model.unload()
    return dict(voltage=voltage, time_axis=time_axis, spike_times=spike_times, t_first=t_first,
                t_first_kc=t_first_kc, vrest=vrest, vthresh=vthresh, dt=dt, spike_count=spike_count)


def _col(rows, name):
    return np.array([float(r[name]) if r[name] not in ("", "nan") else np.nan for r in rows])


def _bool_col(rows, name):
    return np.array([r[name] in ("True", "1", "true") for r in rows])


def load_csv_columns():
    rows = list(csv.DictReader(open(CSV_PATH)))
    counts = np.array([int(r["spike_count"]) for r in rows])
    shift_deg = _col(rows, "shift_deg")
    peak_slope = _col(rows, "peak_slope_mV_per_ms")
    rise_duration = _col(rows, "rise_duration_ms")
    rise_slope = _col(rows, "rise_slope_mV_per_ms")
    max_slope = _col(rows, "max_slope_mV_per_ms")
    mean_voltage_rise = _col(rows, "mean_voltage_rise_mV")
    t_first_from_sim_start = _col(rows, "t_first_spike_from_sim_start_ms")
    first_kc_to_mbon = _col(rows, "first_kc_to_mbon_spike_ms")
    peak_slope_2 = _col(rows, "peak_slope_2_mV_per_ms")
    rise_duration_2 = _col(rows, "rise_duration_2_ms")
    rise_slope_2 = _col(rows, "rise_slope_2_mV_per_ms")
    max_slope_2 = _col(rows, "max_slope_2_mV_per_ms")
    second_spike_no_rise = _bool_col(rows, "second_spike_no_rise")
    peak_slope_3 = _col(rows, "peak_slope_3_mV_per_ms")
    rise_duration_3 = _col(rows, "rise_duration_3_ms")
    rise_slope_3 = _col(rows, "rise_slope_3_mV_per_ms")
    max_slope_3 = _col(rows, "max_slope_3_mV_per_ms")
    third_spike_no_rise = _bool_col(rows, "third_spike_no_rise")
    first_isi = _col(rows, "first_isi_ms")
    second_isi = _col(rows, "second_isi_ms")
    total_current = _col(rows, "total_kc_mbon_current")
    mean_isi = _col(rows, "mean_isi_ms")
    with np.errstate(divide="ignore", invalid="ignore"):
        spike_rate_hz = np.where(mean_isi > 0, 1000.0 / mean_isi, np.nan)
    return dict(counts=counts, shift_deg=shift_deg, peak_slope=peak_slope,
                rise_duration=rise_duration, rise_slope=rise_slope, rate=spike_rate_hz,
                max_slope=max_slope, mean_voltage_rise=mean_voltage_rise,
                t_first_from_sim_start=t_first_from_sim_start,
                first_kc_to_mbon=first_kc_to_mbon,
                peak_slope_2=peak_slope_2,
                rise_duration_2=rise_duration_2, rise_slope_2=rise_slope_2,
                max_slope_2=max_slope_2, second_spike_no_rise=second_spike_no_rise,
                peak_slope_3=peak_slope_3,
                rise_duration_3=rise_duration_3, rise_slope_3=rise_slope_3,
                max_slope_3=max_slope_3, third_spike_no_rise=third_spike_no_rise,
                first_isi=first_isi, second_isi=second_isi, total_current=total_current)


def _saturating_exp(x, A, B, C):
    """Esponenziale saturante: tende ad A per x->+inf (C>0), con B che ne
    fissa il segno/ampiezza -- copre sia relazioni crescenti-saturanti
    (B<0, es. pendenze) sia decrescenti-saturanti (B>0, es. durate/tempi)."""
    return A + B * np.exp(-C * x)


def fit_saturating_exp(x, y):
    """Fit ai minimi quadrati non lineare; ritorna (popt, r2) o None se il
    fit non converge o non ci sono abbastanza punti/variabilita'."""
    if len(x) < 4 or np.std(x) == 0:
        return None
    x_range = x.max() - x.min()
    i_max, i_min = np.argmax(x), np.argmin(x)
    A0 = y[i_max]
    B0 = y[i_min] - A0
    C0 = 1.0 / x_range if x_range > 0 else 1.0
    try:
        popt, _ = curve_fit(_saturating_exp, x, y, p0=[A0, B0, C0],
                            bounds=([-np.inf, -np.inf, 1e-8], [np.inf, np.inf, np.inf]),
                            maxfev=20000)
    except Exception:
        return None
    y_pred = _saturating_exp(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return popt, r2


def draw_results_scatter(ax, x, y, mask, xlabel):
    x_m, y_m = x[mask], y[mask]
    ax.scatter(x_m, y_m, s=22, color=POPULATION_COLORS["MBON"],
               edgecolor="black", linewidth=0.3, alpha=0.6)

    fit = fit_saturating_exp(x_m, y_m)
    if fit is not None:
        popt, r2 = fit
        x_line = np.linspace(x_m.min(), x_m.max(), 200)
        ax.plot(x_line, _saturating_exp(x_line, *popt), color=COLORS["start"],
                linewidth=2.0, zorder=5, label="fit esponenziale saturante")
        title = f"R²={r2:.2f}, n={int(mask.sum())}"
        ax.legend(loc="best", frameon=True, framealpha=0.9, edgecolor="0.3", fontsize=7)
    else:
        r2 = float("nan")
        title = f"n={int(mask.sum())}"
    style_axes(ax, xlabel=xlabel, ylabel=RATE_YLABEL, title=title)
    return r2


def save_metric_figure(folder, filename, def_fn, xlabel, x_all, mask):
    fig, (ax_def, ax_res) = new_figure("error_vs_x", ncols=2)
    def_fn(ax_def)
    draw_results_scatter(ax_res, x_all, DATA["rate"], mask, xlabel)
    fig.tight_layout()
    save_figure(fig, os.path.join(folder, filename))


def main():
    apply_style()
    os.makedirs(DIR_1ST, exist_ok=True)
    os.makedirs(DIR_2ND, exist_ok=True)
    os.makedirs(DIR_3RD, exist_ok=True)

    print("Rigenero la prova di esempio per i pannelli di definizione...")
    ex = get_example_trace()
    voltage, time_axis, t_first = ex["voltage"], ex["time_axis"], ex["t_first"]
    t_first_kc = ex["t_first_kc"]
    vrest, vthresh, dt = ex["vrest"], ex["vthresh"], ex["dt"]
    print(f"Esempio: frame {EXAMPLE_FRAME_ID}, shift {EXAMPLE_SHIFT_DEG}°, "
          f"{ex['spike_count']} spike totali, primo spike a {t_first:.2f} ms")

    eps = 0.05
    above = np.where(voltage > vrest + eps)[0]
    onset_idx = int(above[0]) - 1
    onset_time_val = float(time_axis[onset_idx])
    rise_duration_val = t_first - onset_time_val

    peak_at_first_idx = min(int(np.searchsorted(time_axis, t_first)), len(voltage) - 1)
    rise_slope_val = float((voltage[peak_at_first_idx] - voltage[onset_idx]) / rise_duration_val)
    peak_slope_val = rise_slope_val
    t_peak_a, t_peak_b = onset_time_val, t_first
    v_peak_a, v_peak_b = voltage[onset_idx], voltage[peak_at_first_idx]

    mask_max = (time_axis >= onset_time_val) & (time_axis <= t_first)
    idxs_max = np.nonzero(mask_max)[0]
    dv_max = np.diff(voltage[idxs_max]) / dt
    max_idx_local = int(np.argmax(dv_max))
    max_idx = int(idxs_max[max_idx_local])
    max_slope_val = float(dv_max[max_idx_local])
    t_max_a, t_max_b = time_axis[max_idx], time_axis[max_idx + 1]
    v_max_a, v_max_b = voltage[max_idx], voltage[max_idx + 1]

    integral_val = float(np.trapezoid(voltage[idxs_max], time_axis[idxs_max]))
    mean_voltage_rise_val = integral_val / rise_duration_val

    zoom_start = onset_time_val - 1.5
    zoom_end = t_first + 2.5
    zoom_mask = (time_axis >= zoom_start) & (time_axis <= zoom_end)
    zoom_mask_sim0 = (time_axis >= 0) & (time_axis <= zoom_end)

    sorted_ex_spikes = np.sort(ex["spike_times"])
    t_second = float(sorted_ex_spikes[1])

    # onset del 2° spike -- ricerca vincolata a (t_first, t_second], altrimenti
    # in prove con molti spike si rischia di agganciare per errore la rampa
    # di uno spike successivo (bug scoperto e corretto in mbon_voltage_profile.py)
    mask_onset2 = (time_axis > t_first) & (time_axis <= t_second)
    idxs_onset2 = np.nonzero(mask_onset2)[0]
    above2 = np.where(voltage[idxs_onset2] > vrest + eps)[0]
    onset2_idx = int(idxs_onset2[above2[0]]) - 1
    onset2_time_val = float(time_axis[onset2_idx])
    rise_duration2_val = t_second - onset2_time_val

    peak_at_second_idx = min(int(np.searchsorted(time_axis, t_second)), len(voltage) - 1)
    rise_slope2_val = float((voltage[peak_at_second_idx] - voltage[onset2_idx]) / rise_duration2_val)
    peak_slope2_val = rise_slope2_val
    t_peak2_a, t_peak2_b = onset2_time_val, t_second
    v_peak2_a, v_peak2_b = voltage[onset2_idx], voltage[peak_at_second_idx]

    mask_max2 = (time_axis >= onset2_time_val) & (time_axis <= t_second)
    idxs_max2 = np.nonzero(mask_max2)[0]
    dv_max2 = np.diff(voltage[idxs_max2]) / dt
    max2_idx_local = int(np.argmax(dv_max2))
    max2_idx = int(idxs_max2[max2_idx_local])
    max_slope2_val = float(dv_max2[max2_idx_local])
    t_max2_a, t_max2_b = time_axis[max2_idx], time_axis[max2_idx + 1]
    v_max2_a, v_max2_b = voltage[max2_idx], voltage[max2_idx + 1]

    zoom_end2 = t_second + 1.5
    zoom_mask2 = (time_axis >= zoom_start) & (time_axis <= zoom_end2)

    # onset del 3° spike -- stessa logica, vincolata a (t_second, t_third]
    t_third = float(sorted_ex_spikes[2])
    mask_onset3 = (time_axis > t_second) & (time_axis <= t_third)
    idxs_onset3 = np.nonzero(mask_onset3)[0]
    above3 = np.where(voltage[idxs_onset3] > vrest + eps)[0]
    onset3_idx = int(idxs_onset3[above3[0]]) - 1
    onset3_time_val = float(time_axis[onset3_idx])
    rise_duration3_val = t_third - onset3_time_val

    peak_at_third_idx = min(int(np.searchsorted(time_axis, t_third)), len(voltage) - 1)
    rise_slope3_val = float((voltage[peak_at_third_idx] - voltage[onset3_idx]) / rise_duration3_val)
    peak_slope3_val = rise_slope3_val
    t_peak3_a, t_peak3_b = onset3_time_val, t_third
    v_peak3_a, v_peak3_b = voltage[onset3_idx], voltage[peak_at_third_idx]

    mask_max3 = (time_axis >= onset3_time_val) & (time_axis <= t_third)
    idxs_max3 = np.nonzero(mask_max3)[0]
    dv_max3 = np.diff(voltage[idxs_max3]) / dt
    max3_idx_local = int(np.argmax(dv_max3))
    max3_idx = int(idxs_max3[max3_idx_local])
    max_slope3_val = float(dv_max3[max3_idx_local])
    t_max3_a, t_max3_b = time_axis[max3_idx], time_axis[max3_idx + 1]
    v_max3_a, v_max3_b = voltage[max3_idx], voltage[max3_idx + 1]

    zoom_end3 = t_third + 1.5
    zoom_mask3 = (time_axis >= zoom_start) & (time_axis <= zoom_end3)

    print("Carico i risultati dal CSV...")
    global DATA
    DATA = load_csv_columns()
    has_rate = ~np.isnan(DATA["rate"])
    print(f"Spike rate calcolabile (>=2 spike) su {int(has_rate.sum())}/{len(DATA['rate'])} prove")

    def base_def_panel(ax):
        ax.plot(time_axis[zoom_mask], voltage[zoom_mask], color=POPULATION_COLORS["MBON"], linewidth=1.8)
        ax.axhline(vrest, color=COLORS["mean_reference"], linestyle=":", linewidth=1.0)
        ax.axhline(vthresh, color=COLORS["mean_reference"], linestyle="--", linewidth=1.0)
        ax.axvline(t_first, color=COLORS["actual"], linestyle=":", linewidth=1.2, alpha=0.8)
        ax.text(zoom_end, vrest, "Vrest ", va="bottom", ha="right", fontsize=7, color=COLORS["mean_reference"])
        ax.text(zoom_end, vthresh, "Vthresh ", va="bottom", ha="right", fontsize=7,
                color=COLORS["mean_reference"])
        ax.set_ylim(top=vthresh + 1.5)
        ax.set_xlabel("Tempo [ms]")
        ax.set_ylabel("V [mV]")

    def base_def_panel_2(ax):
        ax.plot(time_axis[zoom_mask2], voltage[zoom_mask2], color=POPULATION_COLORS["MBON"], linewidth=1.8)
        ax.axhline(vrest, color=COLORS["mean_reference"], linestyle=":", linewidth=1.0)
        ax.axhline(vthresh, color=COLORS["mean_reference"], linestyle="--", linewidth=1.0)
        ax.axvline(t_first, color=COLORS["actual"], linestyle=":", linewidth=1.0, alpha=0.5)
        ax.axvline(t_second, color=COLORS["actual"], linestyle=":", linewidth=1.2, alpha=0.8)
        ax.text(zoom_end2, vrest, "Vrest ", va="bottom", ha="right", fontsize=7, color=COLORS["mean_reference"])
        ax.text(zoom_end2, vthresh, "Vthresh ", va="bottom", ha="right", fontsize=7,
                color=COLORS["mean_reference"])
        ax.set_ylim(top=vthresh + 1.5)
        ax.set_xlabel("Tempo [ms]")
        ax.set_ylabel("V [mV]")

    def base_def_panel_3(ax):
        ax.plot(time_axis[zoom_mask3], voltage[zoom_mask3], color=POPULATION_COLORS["MBON"], linewidth=1.8)
        ax.axhline(vrest, color=COLORS["mean_reference"], linestyle=":", linewidth=1.0)
        ax.axhline(vthresh, color=COLORS["mean_reference"], linestyle="--", linewidth=1.0)
        ax.axvline(t_first, color=COLORS["actual"], linestyle=":", linewidth=1.0, alpha=0.35)
        ax.axvline(t_second, color=COLORS["actual"], linestyle=":", linewidth=1.0, alpha=0.5)
        ax.axvline(t_third, color=COLORS["actual"], linestyle=":", linewidth=1.2, alpha=0.8)
        ax.text(zoom_end3, vrest, "Vrest ", va="bottom", ha="right", fontsize=7, color=COLORS["mean_reference"])
        ax.text(zoom_end3, vthresh, "Vthresh ", va="bottom", ha="right", fontsize=7,
                color=COLORS["mean_reference"])
        ax.set_ylim(top=vthresh + 1.5)
        ax.set_xlabel("Tempo [ms]")
        ax.set_ylabel("V [mV]")

    # ══════════════════════ 1° SPIKE ═══════════════════════════════════════

    # ── peak_slope / rise_slope (onset → picco): stessa corda ─────────────
    def def_chord_slope(ax, value):
        base_def_panel(ax)
        ax.plot([onset_time_val, t_first], [voltage[onset_idx], voltage[peak_at_first_idx]],
                color=COLORS["start"], linewidth=1.8, solid_capstyle="butt", zorder=5)
        ax.annotate(
            f"{value:.1f} mV/ms",
            xy=((onset_time_val + t_first) / 2, (voltage[onset_idx] + voltage[peak_at_first_idx]) / 2),
            xytext=(t_peak_a - 4.5, (voltage[onset_idx] + voltage[peak_at_first_idx]) / 2 - 3.0),
            fontsize=8, color=COLORS["start"],
            arrowprops=dict(arrowstyle="->", color=COLORS["start"], linewidth=1.0),
        )

    def def_peak_slope(ax):
        def_chord_slope(ax, peak_slope_val)
        ax.set_title("Pendenza di picco (onset → picco)")

    save_metric_figure(DIR_1ST, "metric_peak_slope.png", def_peak_slope,
                        "Pendenza di picco [mV/ms]", DATA["peak_slope"], has_rate)

    # ── rise_duration ─────────────────────────────────────────────────────
    def def_rise_duration(ax):
        base_def_panel(ax)
        ax.axvline(onset_time_val, color=COLORS["secondary"], linestyle="-.", linewidth=1.4)
        y_arrow = vrest + 1.5
        ax.annotate("", xy=(t_first, y_arrow), xytext=(onset_time_val, y_arrow),
                    arrowprops=dict(arrowstyle="<->", color=COLORS["secondary"], linewidth=1.6))
        ax.text((onset_time_val + t_first) / 2, y_arrow + 0.8, f"{rise_duration_val:.2f} ms",
                ha="center", fontsize=8, color=COLORS["secondary"])
        ax.set_title("Durata rampa (onset → 1° spike)")

    has_rise = ~np.isnan(DATA["rise_duration"])
    rise_and_rate = has_rise & has_rate
    save_metric_figure(DIR_1ST, "metric_rise_duration.png", def_rise_duration,
                        "Durata rampa [ms]", DATA["rise_duration"], rise_and_rate)

    # ── rise_slope: stessa corda di peak_slope ─────────────────────────────
    def def_rise_slope(ax):
        def_chord_slope(ax, rise_slope_val)
        ax.set_title("Pendenza onset → picco 1° spike")

    save_metric_figure(DIR_1ST, "metric_rise_slope.png", def_rise_slope,
                        "Pendenza onset→picco [mV/ms]", DATA["rise_slope"], has_rate)

    # ── max_slope (gradino più ripido nella rampa) ─────────────────────────
    def def_max_slope(ax):
        base_def_panel(ax)
        ax.plot([t_max_a, t_max_b], [v_max_a, v_max_b], color=COLORS["start"],
                linewidth=1.8, solid_capstyle="butt", zorder=5)
        ax.annotate(
            f"{max_slope_val:.1f} mV/ms",
            xy=((t_max_a + t_max_b) / 2, (v_max_a + v_max_b) / 2),
            xytext=(t_max_a - 4.5, (v_max_a + v_max_b) / 2 - 3.0),
            fontsize=8, color=COLORS["start"],
            arrowprops=dict(arrowstyle="->", color=COLORS["start"], linewidth=1.0),
        )
        ax.set_title("Pendenza del tratto più ripido")

    save_metric_figure(DIR_1ST, "metric_max_slope.png", def_max_slope,
                        "Pendenza tratto più ripido [mV/ms]", DATA["max_slope"], has_rate)

    # ── mean_voltage_rise: integrale della tensione (onset → 1° spike) / durata ──
    def def_mean_voltage_rise(ax):
        base_def_panel(ax)
        ax.fill_between(time_axis[mask_max], voltage[mask_max], vrest,
                         color=COLORS["start"], alpha=0.25, zorder=1)
        ax.axhline(mean_voltage_rise_val, color=COLORS["start"], linestyle="--", linewidth=1.2)
        ax.text(t_first + 0.3, mean_voltage_rise_val, f"{mean_voltage_rise_val:.1f} mV",
                va="center", fontsize=8, color=COLORS["start"])
        ax.set_title("Tensione media durante la rampa (integrale / durata)")

    has_mean_v_rise = ~np.isnan(DATA["mean_voltage_rise"])
    save_metric_figure(DIR_1ST, "metric_mean_voltage_rise.png", def_mean_voltage_rise,
                        "Tensione media rampa [mV]", DATA["mean_voltage_rise"],
                        has_mean_v_rise & has_rate)

    # ── tempo al 1° spike dall'inizio della simulazione (t=0) ──────────────
    def def_first_spike_sim0(ax):
        ax.plot(time_axis[zoom_mask_sim0], voltage[zoom_mask_sim0], color=POPULATION_COLORS["MBON"],
                linewidth=1.8)
        ax.axhline(vrest, color=COLORS["mean_reference"], linestyle=":", linewidth=1.0)
        ax.axhline(vthresh, color=COLORS["mean_reference"], linestyle="--", linewidth=1.0)
        ax.axvline(t_first, color=COLORS["actual"], linestyle=":", linewidth=1.2, alpha=0.8)
        ax.text(zoom_end, vrest, "Vrest ", va="bottom", ha="right", fontsize=7, color=COLORS["mean_reference"])
        ax.text(zoom_end, vthresh, "Vthresh ", va="bottom", ha="right", fontsize=7,
                color=COLORS["mean_reference"])
        ax.set_ylim(top=vthresh + 1.5)
        y_arrow = vrest + 1.5
        ax.annotate("", xy=(t_first, y_arrow), xytext=(0.0, y_arrow),
                    arrowprops=dict(arrowstyle="<->", color=COLORS["secondary"], linewidth=1.6))
        ax.text(t_first / 2, y_arrow + 0.8, f"{t_first:.2f} ms",
                ha="center", fontsize=8, color=COLORS["secondary"])
        ax.set_xlabel("Tempo [ms]")
        ax.set_ylabel("V [mV]")
        ax.set_title("Tempo al 1° spike (da t=0)")

    t_first_sim0_and_rate = ~np.isnan(DATA["t_first_from_sim_start"]) & has_rate
    save_metric_figure(DIR_1ST, "metric_first_spike_time_from_sim_start.png", def_first_spike_sim0,
                        "Tempo al 1° spike da t=0 [ms]", DATA["t_first_from_sim_start"],
                        t_first_sim0_and_rate)

    # ── tempo dal 1° spike KC al 1° spike MBON ──────────────────────────────
    def def_first_kc_to_mbon(ax):
        ax.plot(time_axis[zoom_mask_sim0], voltage[zoom_mask_sim0], color=POPULATION_COLORS["MBON"],
                linewidth=1.8)
        ax.axhline(vrest, color=COLORS["mean_reference"], linestyle=":", linewidth=1.0)
        ax.axhline(vthresh, color=COLORS["mean_reference"], linestyle="--", linewidth=1.0)
        ax.axvline(t_first, color=COLORS["actual"], linestyle=":", linewidth=1.2, alpha=0.8)
        ax.axvline(t_first_kc, color=COLORS["secondary"], linestyle="-.", linewidth=1.4)
        ax.text(zoom_end, vrest, "Vrest ", va="bottom", ha="right", fontsize=7, color=COLORS["mean_reference"])
        ax.text(zoom_end, vthresh, "Vthresh ", va="bottom", ha="right", fontsize=7,
                color=COLORS["mean_reference"])
        ax.set_ylim(top=vthresh + 1.5)
        y_arrow = vrest + 1.5
        ax.annotate("", xy=(t_first, y_arrow), xytext=(t_first_kc, y_arrow),
                    arrowprops=dict(arrowstyle="<->", color=COLORS["secondary"], linewidth=1.6))
        ax.text((t_first_kc + t_first) / 2, y_arrow + 0.8, f"{t_first - t_first_kc:.2f} ms",
                ha="center", fontsize=8, color=COLORS["secondary"])
        ax.set_xlabel("Tempo [ms]")
        ax.set_ylabel("V [mV]")
        ax.set_title("Tempo dal 1° spike KC al 1° spike MBON")

    first_kc_and_rate = ~np.isnan(DATA["first_kc_to_mbon"]) & has_rate
    save_metric_figure(DIR_1ST, "metric_first_kc_to_mbon_spike.png", def_first_kc_to_mbon,
                        "Tempo 1° KC → 1° MBON [ms]", DATA["first_kc_to_mbon"], first_kc_and_rate)

    # ══════════════════════ 2° SPIKE ═══════════════════════════════════════

    def def_chord_slope_2(ax, value):
        base_def_panel_2(ax)
        ax.plot([onset2_time_val, t_second], [voltage[onset2_idx], voltage[peak_at_second_idx]],
                color=COLORS["start"], linewidth=1.8, solid_capstyle="butt", zorder=5)
        ax.annotate(
            f"{value:.1f} mV/ms",
            xy=((onset2_time_val + t_second) / 2, (voltage[onset2_idx] + voltage[peak_at_second_idx]) / 2),
            xytext=(t_peak2_a - 4.5, (voltage[onset2_idx] + voltage[peak_at_second_idx]) / 2 - 3.0),
            fontsize=8, color=COLORS["start"],
            arrowprops=dict(arrowstyle="->", color=COLORS["start"], linewidth=1.0),
        )

    def def_peak_slope_2(ax):
        def_chord_slope_2(ax, peak_slope2_val)
        ax.set_title("Pendenza di picco (2° spike, onset → picco)")

    has_rate_2 = ~np.isnan(DATA["peak_slope_2"]) & has_rate
    save_metric_figure(DIR_2ND, "metric_peak_slope_2.png", def_peak_slope_2,
                        "Pendenza di picco [mV/ms]", DATA["peak_slope_2"], has_rate_2)

    lateral_only_2 = has_rate_2 & (np.abs(DATA["shift_deg"]) > 90)
    save_metric_figure(DIR_2ND, "metric_peak_slope_2_lateral_only.png", def_peak_slope_2,
                        "Pendenza di picco [mV/ms]", DATA["peak_slope_2"], lateral_only_2)

    def def_rise_duration_2(ax):
        base_def_panel_2(ax)
        ax.axvline(onset2_time_val, color=COLORS["secondary"], linestyle="-.", linewidth=1.4)
        y_arrow = vrest + 1.5
        ax.annotate("", xy=(t_second, y_arrow), xytext=(onset2_time_val, y_arrow),
                    arrowprops=dict(arrowstyle="<->", color=COLORS["secondary"], linewidth=1.6))
        ax.text((onset2_time_val + t_second) / 2, y_arrow + 0.8, f"{rise_duration2_val:.2f} ms",
                ha="center", fontsize=8, color=COLORS["secondary"])
        ax.set_title("Durata rampa (onset → 2° spike)")

    rise2_and_rate = ~np.isnan(DATA["rise_duration_2"]) & has_rate
    save_metric_figure(DIR_2ND, "metric_rise_duration_2.png", def_rise_duration_2,
                        "Durata rampa 2° [ms]", DATA["rise_duration_2"], rise2_and_rate)

    def def_rise_slope_2(ax):
        def_chord_slope_2(ax, rise_slope2_val)
        ax.set_title("Pendenza onset → picco 2° spike")

    rise_slope2_and_rate = ~np.isnan(DATA["rise_slope_2"]) & has_rate
    save_metric_figure(DIR_2ND, "metric_rise_slope_2.png", def_rise_slope_2,
                        "Pendenza onset→picco 2° [mV/ms]", DATA["rise_slope_2"], rise_slope2_and_rate)

    def def_max_slope_2(ax):
        base_def_panel_2(ax)
        ax.plot([t_max2_a, t_max2_b], [v_max2_a, v_max2_b], color=COLORS["start"],
                linewidth=1.8, solid_capstyle="butt", zorder=5)
        ax.annotate(
            f"{max_slope2_val:.1f} mV/ms",
            xy=((t_max2_a + t_max2_b) / 2, (v_max2_a + v_max2_b) / 2),
            xytext=(t_max2_a - 4.5, (v_max2_a + v_max2_b) / 2 - 3.0),
            fontsize=8, color=COLORS["start"],
            arrowprops=dict(arrowstyle="->", color=COLORS["start"], linewidth=1.0),
        )
        ax.set_title("Pendenza del tratto più ripido (2° spike)")

    max_slope2_and_rate = ~np.isnan(DATA["max_slope_2"]) & has_rate
    save_metric_figure(DIR_2ND, "metric_max_slope_2.png", def_max_slope_2,
                        "Pendenza tratto più ripido 2° [mV/ms]", DATA["max_slope_2"], max_slope2_and_rate)

    # ── first_isi: intervallo 1° spike → 2° spike ──────────────────────────
    def def_first_isi(ax):
        base_def_panel_2(ax)
        y_arrow = vrest + 1.5
        ax.annotate("", xy=(t_second, y_arrow), xytext=(t_first, y_arrow),
                    arrowprops=dict(arrowstyle="<->", color=COLORS["secondary"], linewidth=1.6))
        ax.text((t_first + t_second) / 2, y_arrow + 0.8, f"{t_second - t_first:.2f} ms",
                ha="center", fontsize=8, color=COLORS["secondary"])
        ax.set_title("Intervallo 1° spike → 2° spike")

    first_isi_and_rate = ~np.isnan(DATA["first_isi"]) & has_rate
    save_metric_figure(DIR_2ND, "metric_first_isi.png", def_first_isi,
                        "Intervallo 1°→2° spike [ms]", DATA["first_isi"], first_isi_and_rate)

    # ── casi in cui il 2° spike avviene ma la tensione non risulta salita ──
    # (rampa compressa in un solo timestep di simulazione, non risolta dal
    # logging della tensione -- vedi analisi custom_high_rate_low_slope)
    no_rise_2 = DATA["second_spike_no_rise"] & has_rate
    has_rise_2 = (~DATA["second_spike_no_rise"]) & ~np.isnan(DATA["first_isi"]) & has_rate
    n_no_rise_2 = int(no_rise_2.sum())
    n_total_2 = int((~np.isnan(DATA["first_isi"]) & has_rate).sum())
    print(f"2° spike senza salita di tensione rilevabile: {n_no_rise_2}/{n_total_2} prove")

    fig, ax = new_figure("scatter")
    ax.scatter(DATA["first_isi"][has_rise_2], DATA["rate"][has_rise_2], s=22,
               color=POPULATION_COLORS["MBON"], edgecolor="black", linewidth=0.3, alpha=0.6,
               label="rampa 2° spike rilevata")
    ax.scatter(DATA["first_isi"][no_rise_2], DATA["rate"][no_rise_2], s=26,
               color=COLORS["actual"], edgecolor="black", linewidth=0.3, alpha=0.85,
               label="2° spike senza salita rilevabile")
    style_axes(ax, xlabel="Intervallo 1°→2° spike [ms]", ylabel=RATE_YLABEL,
               title=f"Casi con 2° spike \"piatto\" (n={n_no_rise_2}/{n_total_2})")
    ax.legend(loc="best", frameon=True, framealpha=0.9, edgecolor="0.3", fontsize=7)
    fig.tight_layout()
    save_figure(fig, os.path.join(DIR_2ND, "metric_second_spike_no_rise.png"))

    # ══════════════════════ 3° SPIKE ═══════════════════════════════════════

    def def_chord_slope_3(ax, value):
        base_def_panel_3(ax)
        ax.plot([onset3_time_val, t_third], [voltage[onset3_idx], voltage[peak_at_third_idx]],
                color=COLORS["start"], linewidth=1.8, solid_capstyle="butt", zorder=5)
        ax.annotate(
            f"{value:.1f} mV/ms",
            xy=((onset3_time_val + t_third) / 2, (voltage[onset3_idx] + voltage[peak_at_third_idx]) / 2),
            xytext=(t_peak3_a - 4.5, (voltage[onset3_idx] + voltage[peak_at_third_idx]) / 2 - 3.0),
            fontsize=8, color=COLORS["start"],
            arrowprops=dict(arrowstyle="->", color=COLORS["start"], linewidth=1.0),
        )

    def def_peak_slope_3(ax):
        def_chord_slope_3(ax, peak_slope3_val)
        ax.set_title("Pendenza di picco (3° spike, onset → picco)")

    has_rate_3 = ~np.isnan(DATA["peak_slope_3"]) & has_rate
    save_metric_figure(DIR_3RD, "metric_peak_slope_3.png", def_peak_slope_3,
                        "Pendenza di picco [mV/ms]", DATA["peak_slope_3"], has_rate_3)

    def def_rise_duration_3(ax):
        base_def_panel_3(ax)
        ax.axvline(onset3_time_val, color=COLORS["secondary"], linestyle="-.", linewidth=1.4)
        y_arrow = vrest + 1.5
        ax.annotate("", xy=(t_third, y_arrow), xytext=(onset3_time_val, y_arrow),
                    arrowprops=dict(arrowstyle="<->", color=COLORS["secondary"], linewidth=1.6))
        ax.text((onset3_time_val + t_third) / 2, y_arrow + 0.8, f"{rise_duration3_val:.2f} ms",
                ha="center", fontsize=8, color=COLORS["secondary"])
        ax.set_title("Durata rampa (onset → 3° spike)")

    rise3_and_rate = ~np.isnan(DATA["rise_duration_3"]) & has_rate
    save_metric_figure(DIR_3RD, "metric_rise_duration_3.png", def_rise_duration_3,
                        "Durata rampa 3° [ms]", DATA["rise_duration_3"], rise3_and_rate)

    def def_rise_slope_3(ax):
        def_chord_slope_3(ax, rise_slope3_val)
        ax.set_title("Pendenza onset → picco 3° spike")

    rise_slope3_and_rate = ~np.isnan(DATA["rise_slope_3"]) & has_rate
    save_metric_figure(DIR_3RD, "metric_rise_slope_3.png", def_rise_slope_3,
                        "Pendenza onset→picco 3° [mV/ms]", DATA["rise_slope_3"], rise_slope3_and_rate)

    def def_max_slope_3(ax):
        base_def_panel_3(ax)
        ax.plot([t_max3_a, t_max3_b], [v_max3_a, v_max3_b], color=COLORS["start"],
                linewidth=1.8, solid_capstyle="butt", zorder=5)
        ax.annotate(
            f"{max_slope3_val:.1f} mV/ms",
            xy=((t_max3_a + t_max3_b) / 2, (v_max3_a + v_max3_b) / 2),
            xytext=(t_max3_a - 4.5, (v_max3_a + v_max3_b) / 2 - 3.0),
            fontsize=8, color=COLORS["start"],
            arrowprops=dict(arrowstyle="->", color=COLORS["start"], linewidth=1.0),
        )
        ax.set_title("Pendenza del tratto più ripido (3° spike)")

    max_slope3_and_rate = ~np.isnan(DATA["max_slope_3"]) & has_rate
    save_metric_figure(DIR_3RD, "metric_max_slope_3.png", def_max_slope_3,
                        "Pendenza tratto più ripido 3° [mV/ms]", DATA["max_slope_3"], max_slope3_and_rate)

    # ── second_isi: intervallo 2° spike → 3° spike ─────────────────────────
    def def_second_isi(ax):
        base_def_panel_3(ax)
        y_arrow = vrest + 1.5
        ax.annotate("", xy=(t_third, y_arrow), xytext=(t_second, y_arrow),
                    arrowprops=dict(arrowstyle="<->", color=COLORS["secondary"], linewidth=1.6))
        ax.text((t_second + t_third) / 2, y_arrow + 0.8, f"{t_third - t_second:.2f} ms",
                ha="center", fontsize=8, color=COLORS["secondary"])
        ax.set_title("Intervallo 2° spike → 3° spike")

    second_isi_and_rate = ~np.isnan(DATA["second_isi"]) & has_rate
    save_metric_figure(DIR_3RD, "metric_second_isi.png", def_second_isi,
                        "Intervallo 2°→3° spike [ms]", DATA["second_isi"], second_isi_and_rate)

    # ══════════════════════ GENERALI ════════════════════════════════════════

    # ── spike_rate vs spike_count ────────────────────────────────────────
    fig, ax = new_figure("scatter")
    counts = DATA["counts"]
    draw_results_scatter(ax, counts.astype(float), DATA["rate"], has_rate, "Spike totali (40 ms)")
    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, "spike_rate_vs_spike_count.png"))

    # ── spike_rate vs corrente totale KC->MBON ────────────────────────────
    fig, ax = new_figure("scatter")
    draw_results_scatter(ax, DATA["total_current"], DATA["rate"], has_rate,
                          "Corrente totale KC→MBON [a.u.]")
    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, "spike_rate_vs_total_current.png"))

    # ── pendenza max 2° spike vs pendenza max 3° spike ─────────────────────
    mask_23 = ~np.isnan(DATA["max_slope_2"]) & ~np.isnan(DATA["max_slope_3"])
    x23, y23 = DATA["max_slope_2"][mask_23], DATA["max_slope_3"][mask_23]
    fig, ax = new_figure("scatter")
    ax.scatter(x23, y23, s=22, color=POPULATION_COLORS["MBON"],
               edgecolor="black", linewidth=0.3, alpha=0.6)
    fit23 = fit_saturating_exp(x23, y23)
    if fit23 is not None:
        popt23, r2_23 = fit23
        x_line23 = np.linspace(x23.min(), x23.max(), 200)
        ax.plot(x_line23, _saturating_exp(x_line23, *popt23), color=COLORS["start"],
                linewidth=2.0, zorder=5, label="fit esponenziale saturante")
        ax.legend(loc="best", frameon=True, framealpha=0.9, edgecolor="0.3", fontsize=7)
        title23 = f"R²={r2_23:.2f}, n={int(mask_23.sum())}"
    else:
        title23 = f"n={int(mask_23.sum())}"
    style_axes(ax, xlabel="Pendenza max 2° spike [mV/ms]", ylabel="Pendenza max 3° spike [mV/ms]",
               title=title23)
    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, "max_slope_2_vs_max_slope_3.png"))

    print(f"Figure salvate in: {OUTPUT_DIR}")
    print(f"  1° spike: {DIR_1ST}")
    print(f"  2° spike: {DIR_2ND}")
    print(f"  3° spike: {DIR_3RD}")


if __name__ == "__main__":
    main()
