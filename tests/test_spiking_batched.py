"""
Test dell'integrazione batch_size/precompute_features in
insect_nav.spiking.NeuralNetwork (vedi insect_nav/spiking.py e
insect_nav/base.py::testNavigation_batch).

Richiede pygenn -- vanno eseguiti dentro il container distrobox:
    distrobox enter insect-navContainer -- python tests/test_spiking_batched.py
    distrobox enter insect-navContainer -- pytest tests/test_spiking_batched.py -v

Usano la rete di riferimento già allenata, usata per tutto il benchmarking GPU
di questa repo (copia sicura, mai l'originale sotto Desktop/Test Polo Sim):
    /home/emanuele/insect_nav_gpu_bench/refnet_source/parameters.json
Se quel path non esiste (es. macchina diversa da quella di sviluppo), i test
vengono saltati con un messaggio chiaro invece di fallire.

I test che usano batch_size>1 richiedono una GPU reale (GeNN's
single_threaded_cpu backend rifiuta batch_size>1, vedi il ValueError esplicito
testato in test_batch_size_gt1_requires_gpu) e usano
insect_nav.benchmark.gpu_exclusive() per non entrare in conflitto con altri
processi che usano la stessa GPU sulla macchina.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REFNET_PARAMS_PATH = "/home/emanuele/insect_nav_gpu_bench/refnet_source/parameters.json"


def ok(msg):
    print(f"  ok  {msg}")

def fail(msg):
    print(f"  FAIL  {msg}")
    raise AssertionError(msg)

def skip(msg):
    print(f"  SKIP  {msg}")


def _refnet_available():
    return os.path.isfile(REFNET_PARAMS_PATH)


# ─── batch_size / use_gpu guard ────────────────────────────────────────────

def test_batch_size_gt1_requires_gpu():
    print("\n[1/4] batch_size>1 + use_gpu=False -> ValueError esplicito, prima di qualunque build")
    if not _refnet_available():
        skip(f"{REFNET_PARAMS_PATH} non trovato, salto (rete di riferimento della sessione di benchmarking)")
        return

    from insect_nav import NeuralNetwork
    from insect_nav.parameters import load_parameters_from_file

    params = load_parameters_from_file(REFNET_PARAMS_PATH)
    try:
        NeuralNetwork(params, load_net={"pn_kc": True, "kc_mbon": True}, use_gpu=False, batch_size=4)
        fail("doveva sollevare ValueError")
    except ValueError as e:
        assert "batch_size" in str(e) and "use_gpu" in str(e), f"messaggio inatteso: {e}"
        ok(f"ValueError corretto: {e}")


# ─── batch_size=1: comportamento invariato ─────────────────────────────────

def test_batch_size_1_matches_legacy_behavior():
    print("\n[2/4] batch_size=1 (default) -> comportamento identico a prima dell'integrazione")
    if not _refnet_available():
        skip("rete di riferimento non trovata, salto")
        return

    from insect_nav import NeuralNetwork
    from insect_nav.parameters import load_parameters_from_file
    from insect_nav.vision import loadFrame

    params = load_parameters_from_file(REFNET_PARAMS_PATH)
    nn = NeuralNetwork(params, load_net={"pn_kc": True, "kc_mbon": True}, use_gpu=False)
    try:
        assert nn.batch_size == 1, "batch_size di default deve essere 1"
        frame = loadFrame(0, frames_dir=params["trainingDatasetPath"])
        count = nn.test(frame, 0.0)
        assert isinstance(count, int), f"test() a batch_size=1 deve ritornare un int, non {type(count)}"
        ok(f"test() su batch_size=1 ritorna un int ({count} spike MBON), nessuna eccezione")
    finally:
        nn.model.unload()


# ─── train() richiede batch_size == 1 ──────────────────────────────────────

def test_train_requires_batch_size_1():
    print("\n[3/4] train() con batch_size>1 -> ValueError esplicito (rete costruita su GPU)")
    if not _refnet_available():
        skip("rete di riferimento non trovata, salto")
        return

    from insect_nav import NeuralNetwork
    from insect_nav.benchmark import gpu_exclusive
    from insect_nav.parameters import load_parameters_from_file
    from insect_nav.vision import loadFrame

    params = load_parameters_from_file(REFNET_PARAMS_PATH)
    with gpu_exclusive():
        nn = NeuralNetwork(params, load_net={"pn_kc": True, "kc_mbon": True}, use_gpu=True, batch_size=4)
        try:
            frame = loadFrame(0, frames_dir=params["trainingDatasetPath"])
            try:
                nn.train(frame)
                fail("doveva sollevare ValueError")
            except ValueError as e:
                assert "batch_size" in str(e), f"messaggio inatteso: {e}"
                ok(f"ValueError corretto: {e}")
        finally:
            nn.model.unload()


# ─── test() batchato: lista di frame ───────────────────────────────────────

def test_batched_test_returns_list_of_counts():
    print("\n[4/4] test() con lista di frame (batch_size>1) ritorna una lista di conteggi")
    if not _refnet_available():
        skip("rete di riferimento non trovata, salto")
        return

    from insect_nav import NeuralNetwork
    from insect_nav.benchmark import gpu_exclusive
    from insect_nav.parameters import load_parameters_from_file
    from insect_nav.vision import loadFrame

    params = load_parameters_from_file(REFNET_PARAMS_PATH)
    frames = [loadFrame(i, frames_dir=params["trainingDatasetPath"]) for i in range(3)]

    with gpu_exclusive():
        nn = NeuralNetwork(params, load_net={"pn_kc": True, "kc_mbon": True}, use_gpu=True, batch_size=4)
        try:
            counts = nn.test(frames, 0.0, frame_id=[0, 1, 2])
            assert isinstance(counts, list) and len(counts) == 3, f"atteso list di 3, ottenuto {counts}"
            assert all(isinstance(c, int) for c in counts), "ogni conteggio deve essere un int"
            ok(f"test() batchato ritorna 3 conteggi: {counts}")
        finally:
            nn.model.unload()


# ─── runner ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print(" Test insect_nav.spiking (batch_size/precompute_features)")
    print("=" * 50)

    tests = [
        test_batch_size_gt1_requires_gpu,
        test_batch_size_1_matches_legacy_behavior,
        test_train_requires_batch_size_1,
        test_batched_test_returns_list_of_counts,
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
    print(f" Risultato: {passed}/{len(tests)} test superati (falliti: {failed_count})")
    print("=" * 50)
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
