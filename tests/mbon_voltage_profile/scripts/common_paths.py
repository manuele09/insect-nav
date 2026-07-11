"""Percorsi condivisi per gli script di analisi MBON voltage-profile.

Tutti i dati (input copiato dalla rete/traiettoria originali + output
generato) vivono sotto insect-nav/tests/mbon_voltage_profile/ (input/ e
output/ sono gitignored -- vedi insect-nav/.gitignore -- solo questa
cartella scripts/ viene committata). Solo traiettoria di test -- niente
dataset di training."""

import os

TESTS_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PARAMS_PATH = os.path.join(TESTS_BASE, "input", "parameters.json")
TRAJECTORY_PANORAMA_DIR = os.path.join(TESTS_BASE, "input", "trajectory_panorama")

OUTPUT_TRAJECTORY = os.path.join(TESTS_BASE, "output", "trajectory")

TRAJECTORY_MBON_VOLTAGE_PROFILE_DIR = os.path.join(OUTPUT_TRAJECTORY, "mbon_voltage_profile")
TRAJECTORY_ZERO_SPIKE_20MS_DIR = os.path.join(OUTPUT_TRAJECTORY, "mbon_zero_spike_20ms")
