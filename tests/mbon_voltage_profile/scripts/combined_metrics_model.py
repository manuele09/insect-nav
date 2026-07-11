"""
Combina piu' metriche (riferite allo stesso spike: 1°, 2° o 3°) per predire
la spiking rate, invece di usarne una alla volta come in metric_diagram.py.

Per ciascun gruppo di metriche (1°/2°/3° spike):
  1. PCA (su feature standardizzate) -- diagnostica di quanta ridondanza c'e'
     tra le metriche dello stesso spike (spesso derivate dagli stessi tempi
     di spike, quindi correlate tra loro).
  2. Gradient Boosting Regressor -- combina le metriche in modo non lineare
     per predire il rate, con train/test split e feature importance (quali
     metriche contano davvero una volta combinate).

Risultati per gruppo salvati in <cartella_spike>/analisi_combinata/.

Legge da mbon_voltage_trials.csv, scrive sotto insect-nav/tests/mbon_voltage_profile/
(vedi common_paths.py).

Non richiede pygenn (solo csv/numpy/scikit-learn) -- puo' essere eseguito
anche fuori dal container distrobox, purche' mbon_voltage_trials.csv esista
gia' (generato da mbon_voltage_profile.py).
"""

import csv
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common_paths import TRAJECTORY_MBON_VOLTAGE_PROFILE_DIR  # noqa: E402
from insect_nav.plot_style import (                            # noqa: E402
    COLORS,
    POPULATION_COLORS,
    apply_style,
    new_figure,
    save_figure,
    style_axes,
)

OUTPUT_DIR = TRAJECTORY_MBON_VOLTAGE_PROFILE_DIR
CSV_PATH = os.path.join(OUTPUT_DIR, "mbon_voltage_trials.csv")

RANDOM_STATE = 0
TEST_SIZE = 0.2


def _col(rows, name):
    return np.array([float(r[name]) if r[name] not in ("", "nan") else np.nan for r in rows])


def run_analysis(label, feature_cols, feature_labels, out_subdir):
    print(f"\n=== {label} ===")
    rows = list(csv.DictReader(open(CSV_PATH)))
    mean_isi = _col(rows, "mean_isi_ms")
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(mean_isi > 0, 1000.0 / mean_isi, np.nan)

    features = np.column_stack([_col(rows, c) for c in feature_cols])
    valid = ~np.isnan(rate) & ~np.isnan(features).any(axis=1)
    X = features[valid]
    y = rate[valid]
    print(f"Prove valide (rate + tutte le feature definite): {len(y)}/{len(rows)}")

    out_dir = os.path.join(OUTPUT_DIR, out_subdir, "analisi_combinata")
    os.makedirs(out_dir, exist_ok=True)

    # ── PCA: quanta ridondanza c'e' tra le metriche di questo spike ────────
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    pca = PCA()
    pca.fit(X_std)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    print("Varianza spiegata per componente:", np.round(explained, 3))
    print("Cumulativa:", np.round(cumulative, 3))

    fig, ax = new_figure("error_vs_x")
    comp_idx = np.arange(1, len(explained) + 1)
    ax.bar(comp_idx, explained, color=POPULATION_COLORS["MBON"], edgecolor="black",
           linewidth=0.6, alpha=0.85, label="varianza per componente")
    ax.plot(comp_idx, cumulative, color=COLORS["actual"], marker="o", markersize=5,
            linewidth=1.8, label="varianza cumulativa")
    ax.set_xticks(comp_idx)
    style_axes(ax, xlabel="Componente principale", ylabel="Frazione di varianza spiegata",
               title=f"PCA su {len(feature_cols)} metriche standardizzate ({label})")
    ax.legend(loc="best", frameon=True, framealpha=0.9, edgecolor="0.3", fontsize=8)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "pca_explained_variance.png"))

    # ── Gradient Boosting: combinazione non lineare delle metriche ─────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    model = GradientBoostingRegressor(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    print(f"R² train: {r2_train:.3f}, R² test: {r2_test:.3f}")

    fig, ax = new_figure("scatter")
    ax.scatter(y_train, y_pred_train, s=18, color=POPULATION_COLORS["MBON"],
               edgecolor="black", linewidth=0.3, alpha=0.5, label=f"train (R²={r2_train:.2f})")
    ax.scatter(y_test, y_pred_test, s=22, color=COLORS["actual"],
               edgecolor="black", linewidth=0.3, alpha=0.8, label=f"test (R²={r2_test:.2f})")
    lims = [min(y.min(), y_pred_test.min()), max(y.max(), y_pred_test.max())]
    ax.plot(lims, lims, color="0.3", linestyle="--", linewidth=1.2, zorder=1)
    style_axes(ax, xlabel="Spike rate reale [Hz]", ylabel="Spike rate predetta [Hz]",
               title=f"Gradient Boosting, {len(feature_cols)} metriche combinate ({label})")
    ax.legend(loc="best", frameon=True, framealpha=0.9, edgecolor="0.3", fontsize=8)
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "gradient_boosting_predicted_vs_actual.png"))

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    fig, ax = new_figure("error_vs_x")
    ax.barh([feature_labels[i] for i in order][::-1], importances[order][::-1],
            color=POPULATION_COLORS["MBON"], edgecolor="black", linewidth=0.6, alpha=0.85)
    style_axes(ax, xlabel="Importanza (Gradient Boosting)", ylabel="",
               title=f"Importanza delle metriche combinate ({label})")
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, "feature_importance.png"))

    print("Importanza per metrica:")
    for i in order:
        print(f"  {feature_labels[i]}: {importances[i]:.3f}")
    print(f"Figure salvate in: {out_dir}")


def main():
    apply_style()

    run_analysis(
        "1° spike",
        feature_cols=["rise_duration_ms", "peak_slope_mV_per_ms", "max_slope_mV_per_ms",
                      "mean_voltage_rise_mV", "t_first_spike_from_sim_start_ms",
                      "first_kc_to_mbon_spike_ms"],
        feature_labels=["Durata rampa", "Pendenza di picco", "Pendenza max",
                         "Tensione media rampa", "Tempo 1° spike (t=0)", "Tempo 1° KC→MBON"],
        out_subdir="metriche_1_spike",
    )

    run_analysis(
        "2° spike",
        feature_cols=["rise_duration_2_ms", "peak_slope_2_mV_per_ms", "max_slope_2_mV_per_ms",
                      "first_isi_ms"],
        feature_labels=["Durata rampa 2°", "Pendenza di picco 2°", "Pendenza max 2°",
                         "Intervallo 1°→2°"],
        out_subdir="metriche_2_spike",
    )

    run_analysis(
        "3° spike",
        feature_cols=["rise_duration_3_ms", "peak_slope_3_mV_per_ms", "max_slope_3_mV_per_ms",
                      "second_isi_ms"],
        feature_labels=["Durata rampa 3°", "Pendenza di picco 3°", "Pendenza max 3°",
                         "Intervallo 2°→3°"],
        out_subdir="metriche_3_spike",
    )


if __name__ == "__main__":
    main()
