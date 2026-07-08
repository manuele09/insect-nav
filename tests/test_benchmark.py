"""
Test di insect_nav.benchmark — harness di confronto/benchmark condiviso per
le run di riferimento CPU/GPU (nessuna rete neurale reale coinvolta: qui si
testano solo compare_kc_spike_counts, save_run/load_run e gpu_exclusive con
dati sintetici, senza toccare NeuralNetwork/pygenn, cosi' girano anche
sull'host senza il container).

Esecuzione:
    python tests/test_benchmark.py
    pytest tests/test_benchmark.py -v
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from insect_nav.benchmark import compare_kc_spike_counts, gpu_exclusive, load_run, save_run


def ok(msg):
    print(f"  ok  {msg}")

def fail(msg):
    print(f"  FAIL  {msg}")
    raise AssertionError(msg)


def _make_run(frame_ids, kc_counts, best_degree, shift_degrees=None):
    frame_ids = np.array(frame_ids, dtype=np.int64)
    kc_counts = np.array(kc_counts, dtype=np.int64)
    best_degree = np.array(best_degree, dtype=np.float64)
    if shift_degrees is None:
        shift_degrees = np.arange(kc_counts.shape[1], dtype=np.float64)
    return {
        "frame_ids": frame_ids,
        "shift_degrees": shift_degrees,
        "kc_spike_counts": kc_counts,
        "mbon_spike_counts": kc_counts.copy(),
        "best_degree": best_degree,
        "elapsed_seconds": np.float64(1.23),
    }


# ─── compare_kc_spike_counts ───────────────────────────────────────────────

def test_identical_runs_zero_diff_full_match():
    print("\n[1/6] run identiche -> diff nulla, match rate 1.0")
    ref = _make_run([0, 1, 2], [[10, 20], [30, 40], [5, 6]], [9.0, -9.0, 0.0])
    candidate = _make_run([0, 1, 2], [[10, 20], [30, 40], [5, 6]], [9.0, -9.0, 0.0])
    result = compare_kc_spike_counts(ref, candidate)
    assert result["num_common_frames"] == 3
    assert result["missing_in_candidate"] == []
    assert result["missing_in_ref"] == []
    assert result["mean_abs_diff"] == 0.0
    assert result["mean_rel_diff"] == 0.0
    assert result["max_abs_diff"] == 0.0
    assert result["best_degree_exact_match_rate"] == 1.0
    ok("diff nulla e match rate 1.0 su run identiche")


def test_known_differences_computed_correctly():
    print("\n[2/6] differenze note -> mean/max abs diff e match rate corretti")
    ref = _make_run([0, 1], [[10, 10], [20, 20]], [9.0, -9.0])
    candidate = _make_run([0, 1], [[12, 10], [20, 24]], [9.0, 0.0])
    result = compare_kc_spike_counts(ref, candidate)
    # abs diffs: frame0 -> [2, 0], frame1 -> [0, 4] => mean = 6/4 = 1.5, max = 4
    assert result["mean_abs_diff"] == 1.5, result["mean_abs_diff"]
    assert result["max_abs_diff"] == 4.0, result["max_abs_diff"]
    ref_mean = (10 + 10 + 20 + 20) / 4
    assert abs(result["mean_rel_diff"] - 1.5 / ref_mean) < 1e-12
    # best_degree matches on frame 0 only (9.0 == 9.0), differs on frame 1 (-9.0 != 0.0)
    assert result["best_degree_exact_match_rate"] == 0.5
    ok("mean_abs_diff=1.5, max_abs_diff=4.0, rel diff e match rate a 0.5 come atteso")


def test_missing_frames_aligned_and_reported():
    print("\n[3/6] frame_ids non allineati -> intersezione corretta e frame mancanti riportati")
    ref = _make_run([0, 1, 2], [[1, 1], [2, 2], [3, 3]], [0.0, 0.0, 0.0])
    candidate = _make_run([1, 2, 5], [[2, 2], [3, 3], [9, 9]], [0.0, 0.0, 0.0])
    result = compare_kc_spike_counts(ref, candidate)
    assert result["num_common_frames"] == 2
    assert result["missing_in_candidate"] == [0]
    assert result["missing_in_ref"] == [5]
    # common frames (1, 2) are identical between ref and candidate -> zero diff
    assert result["mean_abs_diff"] == 0.0
    assert result["best_degree_exact_match_rate"] == 1.0
    ok("frame comuni = {1, 2}, frame 0 mancante in candidate, frame 5 mancante in ref")


def test_no_common_frames_returns_nan_safely():
    print("\n[4/6] nessun frame comune -> metriche NaN, nessuna eccezione")
    ref = _make_run([0], [[1, 1]], [0.0])
    candidate = _make_run([99], [[1, 1]], [0.0])
    result = compare_kc_spike_counts(ref, candidate)
    assert result["num_common_frames"] == 0
    assert result["missing_in_candidate"] == [0]
    assert result["missing_in_ref"] == [99]
    import math
    assert math.isnan(result["mean_abs_diff"])
    assert math.isnan(result["best_degree_exact_match_rate"])
    ok("nessuna eccezione, metriche riportate come NaN")


# ─── save_run / load_run ────────────────────────────────────────────────────

def test_save_and_load_run_roundtrip():
    print("\n[5/6] save_run/load_run -> round-trip fedele di array sintetici")
    run = _make_run([0, 1, 2], [[1, 2], [3, 4], [5, 6]], [9.0, 0.0, -9.0])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "run.npz")
        save_run(path, **run)
        assert os.path.isfile(path)
        loaded = load_run(path)
        for key in run:
            assert key in loaded, f"chiave mancante dopo il round-trip: {key}"
            np.testing.assert_array_equal(np.asarray(loaded[key]), np.asarray(run[key]))
        result = compare_kc_spike_counts(run, loaded)
        assert result["mean_abs_diff"] == 0.0
        assert result["best_degree_exact_match_rate"] == 1.0
    ok("tutti gli array sopravvivono identici al round-trip save_run/load_run")


# ─── gpu_exclusive ───────────────────────────────────────────────────────────

def test_gpu_exclusive_lock_acquired_and_released():
    print("\n[6/6] gpu_exclusive -> lock acquisito e rilasciato su un lock file temporaneo")
    import fcntl

    with tempfile.TemporaryDirectory() as tmp:
        lock_path = os.path.join(tmp, "gpu.lock")
        open(lock_path, "w").close()

        entered = []
        with gpu_exclusive(lock_path):
            entered.append(True)
            # While held, a non-blocking attempt to acquire the same lock
            # from a second file handle must fail (proves exclusivity).
            probe = open(lock_path, "w")
            try:
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fail("un secondo lock non bloccante non doveva riuscire mentre gpu_exclusive e' attivo")
            except BlockingIOError:
                ok("secondo lock non bloccante correttamente rifiutato durante gpu_exclusive")
            finally:
                probe.close()

        assert entered == [True]

        # After the context exits, the lock must be free again.
        probe2 = open(lock_path, "w")
        fcntl.flock(probe2.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(probe2.fileno(), fcntl.LOCK_UN)
        probe2.close()
        ok("lock rilasciato correttamente dopo l'uscita dal context manager")


# ─── runner ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print(" Test insect_nav.benchmark")
    print("=" * 50)

    tests = [
        test_identical_runs_zero_diff_full_match,
        test_known_differences_computed_correctly,
        test_missing_frames_aligned_and_reported,
        test_no_common_frames_returns_nan_safely,
        test_save_and_load_run_roundtrip,
        test_gpu_exclusive_lock_acquired_and_released,
    ]

    passed, failed_count = 0, 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed_count += 1

    print(f"\n{'='*50}")
    print(f" Risultato: {passed}/{len(tests)} test superati")
    print("=" * 50)
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
