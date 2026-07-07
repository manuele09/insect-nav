"""
Test di insect_nav.tuning.utilities — conversioni vars <-> nome <-> dict
usate da Tuner/create variants per costruire nomi di cartella univoci.

Esecuzione:
    python tests/test_tuning_utilities.py
    pytest tests/test_tuning_utilities.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from insect_nav.tuning.utilities import (
    name_to_vars,
    params_dict_to_vars,
    vars_to_dict,
    vars_to_name,
)


def ok(msg):
    print(f"  ok  {msg}")

def fail(msg):
    print(f"  FAIL  {msg}")
    raise AssertionError(msg)


SPIKING_VARS = [1500, 10, 0.05, 40.0, 0.3, 0.1, 2]
NON_SPIKING_VARS = [0.01, 64, 4]


def test_vars_to_dict_spiking():
    print("\n[1/6] vars_to_dict — spiking")
    d = vars_to_dict(SPIKING_VARS, "spiking")
    assert d == {
        "target_kcs": 1500,
        "pn_kc_fan_in": 10,
        "pn_kc_weight": 0.05,
        "vthresh": 40.0,
        "vertical_weight": 0.3,
        "horizontal_weight": 0.1,
        "train_step": 2,
    }
    ok("dict spiking corretto")


def test_vars_to_dict_non_spiking():
    print("\n[2/6] vars_to_dict — non-spiking")
    d = vars_to_dict(NON_SPIKING_VARS, "infomax")
    assert d == {"lr": 0.01, "ou": 64, "ts": 4}
    ok("dict non-spiking corretto")


def test_vars_to_name_roundtrip_spiking():
    print("\n[3/6] vars_to_name -> name_to_vars round-trip (spiking)")
    name = vars_to_name(SPIKING_VARS, "spiking")
    recovered = name_to_vars(name, "spiking")
    assert recovered == SPIKING_VARS, f"{recovered} != {SPIKING_VARS}"
    ok(f"round-trip ok: {name}")


def test_vars_to_name_roundtrip_non_spiking():
    print("\n[4/6] vars_to_name -> name_to_vars round-trip (non-spiking)")
    name = vars_to_name(NON_SPIKING_VARS, "infomax")
    recovered = name_to_vars(name, "infomax")
    assert recovered == NON_SPIKING_VARS, f"{recovered} != {NON_SPIKING_VARS}"
    ok(f"round-trip ok: {name}")


def test_name_to_vars_invalid_format_raises():
    print("\n[5/6] name_to_vars con formato non valido solleva ValueError")
    try:
        name_to_vars("not_a_valid_name", "spiking")
        fail("atteso ValueError per nome malformato")
    except ValueError:
        ok("ValueError sollevato correttamente")


def test_params_dict_to_vars_roundtrip_and_missing_keys():
    print("\n[6/6] params_dict_to_vars — round-trip e KeyError su chiavi mancanti")
    params = {
        "target_kcs": 1500,
        "PN_KC_FAN_IN": 10,
        "PN_KC_WEIGHT": 0.05,
        "IF_PARAMS": {"Vthresh": 40.0},
        "VERTICAL_WEIGHT": 0.3,
        "HORIZONTAL_WEIGHT": 0.1,
        "train_step": 2,
    }
    recovered = params_dict_to_vars(params, "spiking")
    assert recovered == SPIKING_VARS, f"{recovered} != {SPIKING_VARS}"
    ok("params_dict_to_vars coerente con vars_to_dict")

    incomplete = dict(params)
    del incomplete["PN_KC_FAN_IN"]
    try:
        params_dict_to_vars(incomplete, "spiking")
        fail("atteso KeyError per chiave mancante")
    except KeyError as e:
        assert "PN_KC_FAN_IN" in str(e)
        ok(f"KeyError sollevato correttamente: {e}")


# ─── runner ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print(" Test insect_nav.tuning.utilities")
    print("=" * 50)

    tests = [
        test_vars_to_dict_spiking,
        test_vars_to_dict_non_spiking,
        test_vars_to_name_roundtrip_spiking,
        test_vars_to_name_roundtrip_non_spiking,
        test_name_to_vars_invalid_format_raises,
        test_params_dict_to_vars_roundtrip_and_missing_keys,
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
