"""
Test del modello PerfectMemory di insect_nav.

Cosa viene testato, in ordine:
    1. memoria vuota       — test() ritorna inf
    2. familiarità         — frame allenato ha score < frame mai visto
    3. minimo su più viste — con N training frames si prende sempre il minimo MAE
    4. save / load pesi    — round-trip produce gli stessi score
    5. shift_degree        — frame ruotato produce score diverso

Esecuzione:
    python tests/test_perfect_memory.py
    pytest tests/test_perfect_memory.py -v
"""

import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from insect_nav.config import NetworkConfig
from insect_nav.memory import PerfectMemory

# ─── helpers ─────────────────────────────────────────────────────────────────

def make_frame(seed: int, h: int = 240, w: int = 320) -> np.ndarray:
    """Immagine BGR casuale riproducibile."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def make_params(weights_dir: str) -> NetworkConfig:
    return NetworkConfig(
        WIDTH=40, HEIGHT=8,
        CROP_TOP=20, CROP_BOTTOM=20,
        USE_VERTICAL_DIST=True, USE_HORIZONTAL_DIST=False,
        VERTICAL_WEIGHT=0.0, HORIZONTAL_WEIGHT=0.0,
        weightsPath=weights_dir,
        plotsTrainPath=os.path.join(weights_dir, "plots/train"),
        plotsTestPath=os.path.join(weights_dir, "plots/test"),
        plotsSimulationPath=os.path.join(weights_dir, "plots/sim"),
    )


def ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def fail(msg: str) -> None:
    print(f"  ✗  {msg}")
    raise AssertionError(msg)


# ─── 1. memoria vuota ────────────────────────────────────────────────────────

def test_empty_memory():
    print("\n[1/5] memoria vuota")
    tmp = tempfile.mkdtemp()
    try:
        model = PerfectMemory(make_params(tmp))
        score = model.test(make_frame(0))
        assert score == float("inf"), f"atteso inf, ottenuto {score}"
        ok("memoria vuota → score = inf")
    finally:
        shutil.rmtree(tmp)


# ─── 2. familiarità ──────────────────────────────────────────────────────────

def test_familiarity():
    print("\n[2/5] familiarità")
    tmp = tempfile.mkdtemp()
    try:
        model = PerfectMemory(make_params(tmp))
        known = make_frame(seed=0)
        unknown = make_frame(seed=99)

        model.train(known)

        score_known = model.test(known)
        score_unknown = model.test(unknown)

        assert score_known < score_unknown, (
            f"score frame allenato ({score_known:.4f}) "
            f"deve essere < frame mai visto ({score_unknown:.4f})"
        )
        ok(f"frame allenato: score={score_known:.4f}  <  frame mai visto: score={score_unknown:.4f}")

        # il frame esattamente identico a quello allenato ha MAE ≈ 0
        assert score_known < 1e-6, f"MAE del frame allenato atteso ≈ 0, ottenuto {score_known}"
        ok(f"MAE frame allenato ≈ 0 ({score_known:.2e})")
    finally:
        shutil.rmtree(tmp)


# ─── 3. minimo su più viste ──────────────────────────────────────────────────

def test_minimum_over_views():
    print("\n[3/5] minimo su più viste")
    tmp = tempfile.mkdtemp()
    try:
        model = PerfectMemory(make_params(tmp))
        frames = [make_frame(seed=i) for i in range(5)]
        query = make_frame(seed=2)   # identico al terzo frame

        for f in frames:
            model.train(f)

        score_multi = model.test(query)

        # con un solo modello allenato sul frame identico
        model_single = PerfectMemory(make_params(tmp))
        model_single.train(frames[2])
        score_single = model_single.test(query)

        # il minimo con 5 viste deve essere ≤ il minimo con 1 vista
        assert score_multi <= score_single + 1e-9, (
            f"score con 5 viste ({score_multi:.6f}) "
            f"deve essere ≤ score con 1 vista ({score_single:.6f})"
        )
        ok(f"score con 5 viste ({score_multi:.6f}) ≤ score con 1 vista ({score_single:.6f})")

        # il frame identico ha sempre score ≈ 0 indipendentemente dagli altri
        assert score_multi < 1e-6, f"MAE atteso ≈ 0, ottenuto {score_multi}"
        ok("minimo correttamente trovato tra le 5 viste (MAE ≈ 0)")
    finally:
        shutil.rmtree(tmp)


# ─── 4. save / load pesi ─────────────────────────────────────────────────────

def test_save_load():
    print("\n[4/5] save / load pesi")
    tmp = tempfile.mkdtemp()
    try:
        params = make_params(tmp)
        model = PerfectMemory(params)
        frame_a = make_frame(seed=7)
        frame_b = make_frame(seed=8)
        query = make_frame(seed=55)

        model.train(frame_a)
        model.train(frame_b)
        score_before = model.test(query)
        model.save_weights()

        # ricrea il modello da zero e carica i pesi
        model2 = PerfectMemory(params, load_net=True)
        score_after = model2.test(query)

        assert abs(score_before - score_after) < 1e-6, (
            f"score prima del save: {score_before:.6f}, "
            f"dopo load: {score_after:.6f}"
        )
        ok(f"score identico prima/dopo save+load ({score_before:.6f})")

        assert len(model2.training_views) == 2
        ok(f"numero di viste caricate corretto: {len(model2.training_views)}")
    finally:
        shutil.rmtree(tmp)


# ─── 5. shift_degree ─────────────────────────────────────────────────────────

def test_shift_degree():
    print("\n[5/5] shift_degree")
    tmp = tempfile.mkdtemp()
    try:
        model = PerfectMemory(make_params(tmp))
        frame = make_frame(seed=3)
        model.train(frame)

        score_0 = model.test(frame, shift_degree=0)
        score_90 = model.test(frame, shift_degree=90)

        assert score_90 > score_0, (
            f"shift 90° ({score_90:.4f}) deve dare score > shift 0° ({score_0:.4f})"
        )
        ok(f"shift 0°: score={score_0:.6f}  <  shift 90°: score={score_90:.4f}")
    finally:
        shutil.rmtree(tmp)


# ─── runner ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 54)
    print(" Test PerfectMemory — insect_nav.memory")
    print("=" * 54)

    tests = [
        test_empty_memory,
        test_familiarity,
        test_minimum_over_views,
        test_save_load,
        test_shift_degree,
    ]

    passed, failed_count = 0, 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            failed_count += 1

    print(f"\n{'='*54}")
    print(f" Risultato: {passed}/{len(tests)} test superati", end="")
    print(" ✓" if failed_count == 0 else f"  ({failed_count} falliti)")
    print("=" * 54)
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
