"""
MBON voltage-profile vs spike-rate data generation (traiettoria di test).

Ipotesi da testare: la velocita' di salita della tensione di membrana del
MBON e' informativa sulla spike rate che il MBON emettera' durante la
presentazione. Questo script genera solo i dati grezzi (CSV) -- tutte le
visualizzazioni sono in metric_diagram.py.

Legge/scrive sotto insect-nav/tests/mbon_voltage_profile/ (vedi
common_paths.py) -- copia locale di parameters.json/weights/frame, gitignored.

Metodologia:
  - Rete caricata su CPU (backend single_threaded_cpu, deterministico).
  - PRESENT_TIME_MS portato da 20 a 40 ms *solo nel dict in memoria* (il
    parameters.json su disco non viene mai riscritto).
  - Tutti i 200 frame della traiettoria x scansione angolare completa a
    360° (40 shift, passo 9°) = 8000 prove.

NOTA (in corso di indagine): il backend single_threaded_cpu di GeNN non si
e' rivelato perfettamente riproducibile run-to-run per questa rete, anche a
parita' di codice/parametri/pesi/ordine delle chiamate -- vedi
tests/test_mbon_reset_determinism.py per la parte di non-determinismo gia'
isolata (dipendenza dallo storico delle chiamate entro la stessa run).

Richiede pygenn -- va eseguito dentro il container distrobox:
    distrobox enter insect-navContainer -- python tests/mbon_voltage_profile/scripts/mbon_voltage_profile.py
"""

import csv
import json
import os
import sys
import time

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
from insect_nav.spiking import NeuralNetwork              # noqa: E402
from insect_nav.vision import loadFrame                   # noqa: E402

NEW_PRESENT_TIME_MS = 40.0


def load_params_without_touching_disk(path):
    with open(path) as f:
        parameters = json.load(f)
    return update_paths(parameters, path)


def compute_onset_time(time_axis, voltage, vrest, start_time, end_time, eps=0.05):
    """Ultimo istante ancora fermo a Vrest, subito PRIMA che il MBON si
    scosti misurabilmente (soglia eps in mV), cercando SOLO nella finestra
    (start_time, end_time] -- cioe' prima dello spike di interesse.

    NOTA: la ricerca deve essere limitata a end_time (lo spike a cui si
    riferisce l'onset), altrimenti in prove con molti spike ravvicinati la
    ricerca "sconfina" e trova per errore il campione elevato di uno spike
    SUCCESSIVO (es. il 3o), producendo un onset/durata rampa insensati
    (durata negativa) per lo spike di interesse.
    """
    mask = (time_axis > start_time) & (time_axis <= end_time)
    idxs = np.nonzero(mask)[0]
    if len(idxs) == 0:
        return np.nan
    above = np.where(voltage[idxs] > vrest + eps)[0]
    if len(above) == 0:
        return np.nan
    first_above_idx = idxs[above[0]]
    preceding_idx = first_above_idx - 1
    if preceding_idx < 0:
        return np.nan
    return float(time_axis[preceding_idx])


def compute_mean_voltage_rise(time_axis, voltage, onset_time, peak_time):
    """Integrale della tensione tra onset e picco (spike), diviso per la
    durata della finestra -- tensione media durante la rampa."""
    if np.isnan(onset_time) or np.isnan(peak_time) or peak_time <= onset_time:
        return np.nan
    mask = (time_axis >= onset_time) & (time_axis <= peak_time)
    idxs = np.nonzero(mask)[0]
    if len(idxs) < 2:
        return np.nan
    duration = peak_time - onset_time
    integral = float(np.trapezoid(voltage[idxs], time_axis[idxs]))
    return integral / duration


def compute_rise_slope(time_axis, voltage, onset_time, peak_time):
    """Corda a due estremi onset->picco: (V_picco - V_onset) / durata."""
    if np.isnan(onset_time) or np.isnan(peak_time) or peak_time <= onset_time:
        return np.nan
    onset_idx = min(int(np.searchsorted(time_axis, onset_time)), len(voltage) - 1)
    peak_idx = min(int(np.searchsorted(time_axis, peak_time)), len(voltage) - 1)
    duration = peak_time - onset_time
    return float((voltage[peak_idx] - voltage[onset_idx]) / duration)


def compute_max_slope(time_axis, voltage, dt, onset_time, peak_time):
    """Massimo dV/dt istantaneo nella finestra [onset, picco]."""
    if np.isnan(onset_time) or np.isnan(peak_time) or peak_time <= onset_time:
        return np.nan
    mask = (time_axis >= onset_time) & (time_axis <= peak_time)
    idxs = np.nonzero(mask)[0]
    if len(idxs) < 2:
        return np.nan
    return float(np.max(np.diff(voltage[idxs]) / dt))


def main():
    os.makedirs(TRAJECTORY_MBON_VOLTAGE_PROFILE_DIR, exist_ok=True)

    mtime_before = os.path.getmtime(PARAMS_PATH)

    params = load_params_without_touching_disk(PARAMS_PATH)
    print(f"PRESENT_TIME_MS nel file: {params['PRESENT_TIME_MS']} ms")
    params["PRESENT_TIME_MS"] = NEW_PRESENT_TIME_MS
    print(f"PRESENT_TIME_MS usato per questa run (solo in memoria): {params['PRESENT_TIME_MS']} ms")

    nn = NeuralNetwork(params, load_net={"pn_kc": True, "kc_mbon": True}, use_gpu=False)
    nn.logger.update_config("voltages", {"mbon": True})
    nn.logger.update_config("currents", {"kc_mbon": True})

    vrest = params["LIF_PARAMS"]["Vrest"]
    dt = params["DT"]

    frame_ids = list(range(0, 200))  # tutti i 200 frame della traiettoria
    shift_degrees = list(np.arange(-180, 180, 9))  # scansione completa a 360°, 40 shift
    print(f"Frame campionati: {len(frame_ids)} (tutti)")
    print(f"Shift angolari per frame: {len(shift_degrees)} ({shift_degrees[0]:.0f} .. {shift_degrees[-1]:.0f} deg, 360°)")
    print(f"Totale prove: {len(frame_ids) * len(shift_degrees)}")

    csv_path = os.path.join(TRAJECTORY_MBON_VOLTAGE_PROFILE_DIR, "mbon_voltage_trials.csv")
    rows = []

    t0 = time.time()
    n_done = 0
    for frame_id in frame_ids:
        frame = loadFrame(frame_id, frames_dir=TRAJECTORY_PANORAMA_DIR)
        for shift in shift_degrees:
            spike_count = nn.test(frame, shift_degree=shift)

            v_data = nn.logger.get_voltages("mbon")
            s_data = nn.logger.get_spikes("mbon")
            kc_data = nn.logger.get_spikes("kc")
            c_data = nn.logger.get_currents("kc_mbon")
            voltage = v_data["voltages"][0]
            time_axis = v_data["time_axis"]
            spike_times = s_data["times"]
            kc_spike_times = kc_data["times"]
            # Corrente totale in ingresso al MBON (KC->MBON) durante la
            # presentazione: integrale (somma di Riemann) della corrente
            # post-sinaptica sull'intera finestra di 40 ms.
            total_current = float(np.sum(c_data["currents"][0]) * dt)

            sorted_spikes = np.sort(spike_times)
            t_first = float(sorted_spikes[0]) if spike_count > 0 else np.nan
            t_first_kc = float(np.min(kc_spike_times)) if len(kc_spike_times) > 0 else np.nan
            first_kc_to_mbon = (
                (t_first - t_first_kc) if (spike_count > 0 and not np.isnan(t_first_kc)) else np.nan
            )
            onset_time = (
                compute_onset_time(time_axis, voltage, vrest, start_time=0.0, end_time=t_first)
                if spike_count > 0 else np.nan
            )
            rise_duration = (t_first - onset_time) if (spike_count > 0 and not np.isnan(onset_time)) else np.nan
            rise_slope = compute_rise_slope(time_axis, voltage, onset_time, t_first)
            peak_slope = rise_slope
            max_slope = compute_max_slope(time_axis, voltage, dt, onset_time, t_first)
            mean_voltage_rise = compute_mean_voltage_rise(time_axis, voltage, onset_time, t_first)
            mean_isi = float(np.mean(np.diff(sorted_spikes))) if spike_count >= 2 else np.nan
            mean_rate_hz = spike_count / (NEW_PRESENT_TIME_MS / 1000.0)

            if spike_count >= 2:
                t_second = float(sorted_spikes[1])
                onset_time_2 = compute_onset_time(time_axis, voltage, vrest, start_time=t_first, end_time=t_second)
                rise_duration_2 = (t_second - onset_time_2) if not np.isnan(onset_time_2) else np.nan
                rise_slope_2 = compute_rise_slope(time_axis, voltage, onset_time_2, t_second)
                peak_slope_2 = rise_slope_2
                max_slope_2 = compute_max_slope(time_axis, voltage, dt, onset_time_2, t_second)
                first_isi = t_second - t_first
                second_spike_no_rise = np.isnan(onset_time_2)
            else:
                t_second = onset_time_2 = rise_duration_2 = rise_slope_2 = peak_slope_2 = np.nan
                max_slope_2 = np.nan
                first_isi = np.nan
                second_spike_no_rise = False

            if spike_count >= 3:
                t_third = float(sorted_spikes[2])
                onset_time_3 = compute_onset_time(time_axis, voltage, vrest, start_time=t_second, end_time=t_third)
                rise_duration_3 = (t_third - onset_time_3) if not np.isnan(onset_time_3) else np.nan
                rise_slope_3 = compute_rise_slope(time_axis, voltage, onset_time_3, t_third)
                peak_slope_3 = rise_slope_3
                max_slope_3 = compute_max_slope(time_axis, voltage, dt, onset_time_3, t_third)
                second_isi = t_third - t_second
                third_spike_no_rise = np.isnan(onset_time_3)
            else:
                t_third = onset_time_3 = rise_duration_3 = rise_slope_3 = peak_slope_3 = np.nan
                max_slope_3 = np.nan
                second_isi = np.nan
                third_spike_no_rise = False

            rows.append({
                "frame_id": frame_id,
                "shift_deg": shift,
                "spike_count": spike_count,
                "t_first_spike_from_sim_start_ms": t_first,
                "t_second_spike_from_sim_start_ms": t_second,
                "t_third_spike_from_sim_start_ms": t_third,
                "first_kc_to_mbon_spike_ms": first_kc_to_mbon,
                "rise_duration_ms": rise_duration,
                "rise_slope_mV_per_ms": rise_slope,
                "peak_slope_mV_per_ms": peak_slope,
                "max_slope_mV_per_ms": max_slope,
                "mean_voltage_rise_mV": mean_voltage_rise,
                "rise_duration_2_ms": rise_duration_2,
                "rise_slope_2_mV_per_ms": rise_slope_2,
                "peak_slope_2_mV_per_ms": peak_slope_2,
                "max_slope_2_mV_per_ms": max_slope_2,
                "second_spike_no_rise": second_spike_no_rise,
                "rise_duration_3_ms": rise_duration_3,
                "rise_slope_3_mV_per_ms": rise_slope_3,
                "peak_slope_3_mV_per_ms": peak_slope_3,
                "max_slope_3_mV_per_ms": max_slope_3,
                "third_spike_no_rise": third_spike_no_rise,
                "first_isi_ms": first_isi,
                "second_isi_ms": second_isi,
                "total_kc_mbon_current": total_current,
                "mean_isi_ms": mean_isi,
                "mean_rate_hz": mean_rate_hz,
            })

            n_done += 1

        if (frame_id + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  frame {frame_id}: fatto ({n_done}/{len(frame_ids) * len(shift_degrees)}, "
                  f"{elapsed:.1f}s trascorsi)")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Dati salvati in: {csv_path}")

    mtime_after = os.path.getmtime(PARAMS_PATH)
    assert mtime_before == mtime_after, "parameters.json e' stato toccato! (non doveva succedere)"
    print("Verificato: parameters.json non e' stato modificato.")

    nn.model.unload()


if __name__ == "__main__":
    main()
