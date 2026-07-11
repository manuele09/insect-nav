"""
Test di regressione: NeuralNetwork.test() deve essere indipendente dalla
storia delle chiamate precedenti -- reset_network() (chiamato internamente
da test(), vedi insect_nav/spiking.py) deve riportare la rete in uno stato
tale che presentare lo stesso frame allo stesso shift dia sempre lo stesso
risultato, indipendentemente da quante altre presentazioni sono state fatte
prima sulla stessa istanza di rete.

BUG SCOPERTO (sessione di analisi in tests/mbon_voltage_profile/): questo
NON è vero. Presentare frame_id=36 allo shift -36° subito dopo la creazione
della rete dà 13 spike MBON; presentare lo stesso identico frame/shift dopo
aver già presentato lo stesso frame agli shift precedenti dello scan
(-180°, -171°, ..., -45°) dà invece 14 spike -- e 14 è anche il valore
originariamente registrato nel CSV generato dallo scan completo
(tests/mbon_voltage_profile/output/trajectory/mbon_voltage_profile/
mbon_voltage_trials.csv), che presenta i frame in quell'ordine.

I pesi kc_mbon sono stati verificati bit-identici prima e dopo il warmup
(mod=-1 blocca correttamente la plasticità durante test()), quindi non è un
problema di drift dei pesi: reset_network() lascia non resettato qualche
altro stato della rete. Candidati non ancora esclusi: stato interno di APL,
buffer di spike-recording di GeNN, RNG interno. Vedi la history-dependence
discussa nella sessione mbon_voltage_profile per il contesto completo.

Questo script serve a riprodurre il problema in isolamento e a verificare in
futuro se un fix di reset_network() lo risolve (il test deve passare quando
il bug sarà corretto).

Richiede pygenn -- va eseguito dentro il container distrobox:
    distrobox enter insect-navContainer -- python tests/test_mbon_reset_determinism.py
    distrobox enter insect-navContainer -- pytest tests/test_mbon_reset_determinism.py -v

Usa la rete/traiettoria di test copiate in tests/mbon_voltage_profile/input/
(gitignored, vedi .gitignore -- dati locali della sessione di analisi). Se
assenti (es. macchina diversa da quella della sessione), il test viene
saltato con un messaggio chiaro invece di fallire.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "mbon_voltage_profile", "input")
PARAMS_PATH = os.path.join(TEST_DATA_DIR, "parameters.json")
FRAMES_DIR = os.path.join(TEST_DATA_DIR, "trajectory_panorama")

PRESENT_TIME_MS = 40.0
SHIFT_DEGREES = list(np.arange(-180, 180, 9))

# Prova nota per manifestare il bug (vedi tests/mbon_voltage_profile/):
# frame_id=36, shift=-36 e' la prova a spike rate piu' alta di tutto lo
# scan 200 frame x 40 shift, ed e' quella su cui e' stata scoperta la
# dipendenza dallo storico delle chiamate.
TARGET_FRAME_ID = 36
TARGET_SHIFT_DEG = -36.0


def ok(msg):
    print(f"  ok  {msg}")

def fail(msg):
    print(f"  FAIL  {msg}")
    raise AssertionError(msg)

def skip(msg):
    print(f"  SKIP  {msg}")


def _test_data_available():
    return os.path.isfile(PARAMS_PATH) and os.path.isdir(FRAMES_DIR)


def _load_params():
    from insect_nav.parameters import update_paths
    with open(PARAMS_PATH) as f:
        parameters = json.load(f)
    parameters = update_paths(parameters, PARAMS_PATH)
    parameters["PRESENT_TIME_MS"] = PRESENT_TIME_MS
    return parameters


def _build_network(params):
    from insect_nav.spiking import NeuralNetwork
    nn = NeuralNetwork(params, load_net={"pn_kc": True, "kc_mbon": True}, use_gpu=False)
    nn.logger.update_config("voltages", {"mbon": True})
    return nn


def test_reset_network_is_history_independent():
    print("\n[1/1] test() sullo stesso frame/shift deve dare lo stesso risultato "
          "indipendentemente dalle presentazioni precedenti")
    if not _test_data_available():
        skip(f"dati di test non trovati in {TEST_DATA_DIR}, salto "
             "(rete/traiettoria copiate per la sessione mbon_voltage_profile, "
             "non presenti su questa macchina)")
        return

    from insect_nav.vision import loadFrame

    target_shift_idx = SHIFT_DEGREES.index(TARGET_SHIFT_DEG)
    frame = loadFrame(TARGET_FRAME_ID, frames_dir=FRAMES_DIR)

    # ── A: presentazione isolata, subito dopo la creazione della rete ──────
    params = _load_params()
    nn = _build_network(params)
    try:
        nn.kc_mbon.vars["g"].pull_from_device()
        g_before = nn.kc_mbon.vars["g"].values.copy()

        count_isolated = nn.test(frame, shift_degree=TARGET_SHIFT_DEG)
        v_isolated = nn.logger.get_voltages("mbon")["voltages"][0].copy()
    finally:
        nn.model.unload()

    # ── B: stesso frame/shift, ma dopo aver gia' presentato lo stesso frame
    #        agli shift precedenti dello scan (nessun'altra differenza) ─────
    params = _load_params()
    nn = _build_network(params)
    try:
        for shift in SHIFT_DEGREES[:target_shift_idx]:
            nn.test(frame, shift_degree=shift)

        nn.kc_mbon.vars["g"].pull_from_device()
        g_after = nn.kc_mbon.vars["g"].values.copy()

        count_with_history = nn.test(frame, shift_degree=TARGET_SHIFT_DEG)
        v_with_history = nn.logger.get_voltages("mbon")["voltages"][0].copy()
    finally:
        nn.model.unload()

    weights_changed = not np.allclose(g_before, g_after)
    print(f"  frame_id={TARGET_FRAME_ID}, shift={TARGET_SHIFT_DEG}°")
    print(f"  spike_count isolato:        {count_isolated}")
    print(f"  spike_count dopo storico:   {count_with_history}")
    print(f"  pesi kc_mbon cambiati nel frattempo? {weights_changed}")

    if weights_changed:
        fail("i pesi kc_mbon sono cambiati durante il warmup: il test non isola "
             "correttamente il fenomeno (mod=-1 dovrebbe bloccare la plasticita' "
             "in test())")

    if count_isolated != count_with_history:
        fail(
            f"NON DETERMINISTICO: stesso frame_id/shift ({TARGET_FRAME_ID}/"
            f"{TARGET_SHIFT_DEG}°) da {count_isolated} spike se presentato "
            f"subito dopo la creazione della rete, ma {count_with_history} "
            f"spike se preceduto da {target_shift_idx} presentazioni dello "
            "stesso frame ad altri shift. I pesi kc_mbon sono verificati "
            "invariati: reset_network() lascia non resettato qualche altro "
            "stato della rete (vedi insect_nav/spiking.py, "
            "_create_reset_custom_update / reset_network)."
        )

    if not np.allclose(v_isolated, v_with_history):
        first_diff = int(np.argmax(~np.isclose(v_isolated, v_with_history)))
        fail(
            "spike_count combacia ma le tracce di tensione MBON divergono "
            f"(prima differenza al campione {first_diff}): reset_network() "
            "non riporta la rete in uno stato identico tra una chiamata a "
            "test() e la successiva."
        )

    ok(f"test() e' indipendente dallo storico ({count_isolated} spike in entrambi i casi)")


def main():
    print("=" * 60)
    print(" Test determinismo reset -- NeuralNetwork.test()")
    print("=" * 60)

    tests = [test_reset_network_is_history_independent]
    passed, failed_count = 0, 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed_count += 1

    print(f"\n{'='*60}")
    print(f" Risultato: {passed}/{len(tests)} test superati (falliti: {failed_count})")
    print("=" * 60)
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
