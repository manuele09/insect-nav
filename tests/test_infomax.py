"""
Test del modello Infomax di insect_nav.

Cosa viene testato, in ordine:
    1. weights_shape   — shape dei pesi dopo l'init: (output_units, WIDTH)
    2. weights_init    — i pesi iniziali non sono tutti uguali (std > 0)
    3. train_modifies  — il training cambia i pesi
    4. test_output     — test() ritorna un float valido (non NaN, > 0)
    5. save_load       — round-trip save_weights / load_weights

Esecuzione:
    python tests/test_infomax.py
    pytest tests/test_infomax.py -v
"""

import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from insect_nav.config import NetworkConfig
from insect_nav.infomax import Infomax

# ─── helpers ─────────────────────────────────────────────────────────────────

def make_frame(seed=0, h=240, w=320) -> np.ndarray:
    """Frame BGR sintetico riproducibile."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def make_params(weights_path: str) -> NetworkConfig:
    return NetworkConfig(
        WIDTH=40, HEIGHT=8,
        CROP_TOP=20, CROP_BOTTOM=20,
        USE_VERTICAL_DIST=True, USE_HORIZONTAL_DIST=False,
        VERTICAL_WEIGHT=0.0, HORIZONTAL_WEIGHT=0.0,
        output_units=20,
        learning_rate=0.01,
        weightsPath=weights_path,
        plotsTrainPath=os.path.join(weights_path, "plots", "train"),
        plotsTestPath=os.path.join(weights_path, "plots", "test"),
        plotsSimulationPath=os.path.join(weights_path, "plots", "sim"),
        training_dataset_mean=0.0,
        training_dataset_std=1.0,
    )


def ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def fail(msg: str) -> None:
    print(f"  ✗  {msg}")
    raise AssertionError(msg)


# ─── 1. weights_shape ────────────────────────────────────────────────────────

def test_weights_shape():
    print("\n[1/5] weights_shape")
    tmpdir = tempfile.mkdtemp()
    try:
        params = make_params(tmpdir)
        model = Infomax(params, calculate_mean=False)

        expected = (params["output_units"], params["WIDTH"])
        assert model.weights.shape == expected, \
            f"atteso {expected}, ottenuto {model.weights.shape}"
        ok(f"shape pesi: {model.weights.shape}  (output_units={params['output_units']}, WIDTH={params['WIDTH']})")
    finally:
        shutil.rmtree(tmpdir)


# ─── 2. weights_init ─────────────────────────────────────────────────────────

def test_weights_init():
    print("\n[2/5] weights_init")
    tmpdir = tempfile.mkdtemp()
    try:
        params = make_params(tmpdir)
        model = Infomax(params, calculate_mean=False)

        std = float(model.weights.std())
        assert std > 0, f"std dei pesi iniziali = {std}, atteso > 0"
        ok(f"pesi iniziali non costanti (std={std:.4f})")

        mean = float(model.weights.mean())
        assert abs(mean) < 0.1, f"mean dei pesi iniziali = {mean:.4f}, atteso ≈ 0"
        ok(f"pesi iniziali centrati attorno a 0 (mean={mean:.4f})")
    finally:
        shutil.rmtree(tmpdir)


# ─── 3. train_modifies ───────────────────────────────────────────────────────

def test_train_modifies():
    print("\n[3/5] train_modifies")
    tmpdir = tempfile.mkdtemp()
    try:
        params = make_params(tmpdir)
        model = Infomax(params, calculate_mean=False)

        weights_before = model.weights.copy()
        frame = make_frame(seed=7)
        model.train(frame)

        assert not np.array_equal(model.weights, weights_before), \
            "i pesi non sono cambiati dopo train()"
        ok("train() modifica i pesi")

        # training ripetuto sullo stesso frame continua a cambiare i pesi
        weights_after_1 = model.weights.copy()
        model.train(frame)
        assert not np.array_equal(model.weights, weights_after_1)
        ok("train() ripetuto continua a modificare i pesi")
    finally:
        shutil.rmtree(tmpdir)


# ─── 4. test_output ──────────────────────────────────────────────────────────

def test_test_output():
    print("\n[4/5] test_output")
    tmpdir = tempfile.mkdtemp()
    try:
        params = make_params(tmpdir)
        model = Infomax(params, calculate_mean=False)

        frame = make_frame(seed=3)
        score = model.test(frame, shift_degree=0)

        assert isinstance(score, float), \
            f"test() deve ritornare float, ottenuto {type(score)}"
        ok(f"test() ritorna float: {score:.4f}")

        assert not np.isnan(score), "test() ritorna NaN"
        ok("score non è NaN")

        assert score > 0, f"score atteso > 0, ottenuto {score}"
        ok(f"score > 0 (={score:.4f})")

        # shift diverso → score diverso
        score_shifted = model.test(frame, shift_degree=45)
        assert score != score_shifted, \
            "shift_degree=0 e shift_degree=45 producono lo stesso score"
        ok(f"shift=0° ({score:.4f}) ≠ shift=45° ({score_shifted:.4f})")
    finally:
        shutil.rmtree(tmpdir)


# ─── 5. save_load ────────────────────────────────────────────────────────────

def test_save_load():
    print("\n[5/5] save_load")
    tmpdir = tempfile.mkdtemp()
    try:
        params = make_params(tmpdir)
        model = Infomax(params, calculate_mean=False)

        # allenamento per avere pesi non banali
        for seed in range(5):
            model.train(make_frame(seed=seed))

        weights_original = model.weights.copy()
        model.save_weights()

        # carica in un nuovo modello
        params2 = make_params(tmpdir)
        model2 = Infomax(params2, load_net=True, calculate_mean=False)

        assert np.array_equal(model2.weights, weights_original), \
            "i pesi caricati differiscono da quelli salvati"
        ok("save_weights / load_weights: pesi identici dopo il round-trip")

        # i due modelli producono lo stesso score sullo stesso frame
        frame = make_frame(seed=99)
        score1 = model.test(frame)
        score2 = model2.test(frame)
        assert score1 == score2, \
            f"score diverge dopo load: {score1} vs {score2}"
        ok(f"test() produce lo stesso score dopo load ({score1:.4f})")
    finally:
        shutil.rmtree(tmpdir)


# ─── runner ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print(" Test modello Infomax — insect_nav.infomax")
    print("=" * 50)

    tests = [
        test_weights_shape,
        test_weights_init,
        test_train_modifies,
        test_test_output,
        test_save_load,
    ]

    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  ✗  {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f" Risultato: {passed}/{len(tests)} test superati", end="")
    print(" ✓" if failed == 0 else f"  ({failed} falliti)")
    print("=" * 50)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
