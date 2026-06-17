"""
Test di NetworkConfig — interfaccia dict-compatibile e serializzazione JSON.

Cosa viene testato, in ordine:
    1. valori default
    2. istanziazione con kwargs
    3. accesso dict cfg["KEY"]
    4. assegnazione dict cfg["KEY"] = val
    5. cfg.get(key, default)
    6. operatore in
    7. round-trip to_json / from_json
    8. chiavi sconosciute nel JSON ignorate da from_json
    9. cfg.items() copre tutti i campi

Esecuzione:
    python tests/test_config.py
    pytest tests/test_config.py -v
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataclasses import fields
from insect_nav.config import NetworkConfig


def ok(msg):
    print(f"  ✓  {msg}")

def fail(msg):
    print(f"  ✗  {msg}")
    raise AssertionError(msg)


# ─── 1. valori default ────────────────────────────────────────────────────────

def test_defaults():
    print("\n[1/9] valori default")
    cfg = NetworkConfig()

    assert cfg.WIDTH == 40,           f"WIDTH atteso 40, ottenuto {cfg.WIDTH}"
    assert cfg.HEIGHT == 8,           f"HEIGHT atteso 8, ottenuto {cfg.HEIGHT}"
    assert cfg.NUM_KC == 2000,        f"NUM_KC atteso 2000, ottenuto {cfg.NUM_KC}"
    assert cfg.train_step == 1,       f"train_step atteso 1, ottenuto {cfg.train_step}"
    assert cfg.USE_VERTICAL_DIST is True
    assert cfg.USE_HORIZONTAL_DIST is False
    assert cfg.training_dataset_mean is None
    ok("tutti i campi default hanno i valori attesi")


# ─── 2. istanziazione con kwargs ─────────────────────────────────────────────

def test_kwargs():
    print("\n[2/9] istanziazione con kwargs")
    cfg = NetworkConfig(NUM_KC=500, WIDTH=20, name="my_net")

    assert cfg.NUM_KC == 500
    assert cfg.WIDTH == 20
    assert cfg.name == "my_net"
    assert cfg.HEIGHT == 8     # invariato
    ok("i kwargs sovrascrivono i default, gli altri restano invariati")


# ─── 3. accesso dict cfg["KEY"] ───────────────────────────────────────────────

def test_getitem():
    print("\n[3/9] accesso dict cfg[\"KEY\"]")
    cfg = NetworkConfig(NUM_KC=1234)

    assert cfg["NUM_KC"] == 1234
    assert cfg["WIDTH"] == 40
    assert cfg["name"] == "network"

    try:
        _ = cfg["chiave_inesistente"]
        fail("attesa KeyError per chiave inesistente")
    except KeyError:
        ok("KeyError per chiave inesistente")

    ok("cfg[\"KEY\"] restituisce il valore corretto")


# ─── 4. assegnazione dict cfg["KEY"] = val ────────────────────────────────────

def test_setitem():
    print("\n[4/9] assegnazione dict cfg[\"KEY\"] = val")
    cfg = NetworkConfig()

    cfg["NUM_KC"] = 999
    assert cfg.NUM_KC == 999
    assert cfg["NUM_KC"] == 999
    ok("cfg[\"NUM_KC\"] = 999 → cfg.NUM_KC == 999")

    cfg["INPUT_SCALE"] = 0.123
    assert abs(cfg.INPUT_SCALE - 0.123) < 1e-9
    ok("cfg[\"INPUT_SCALE\"] = 0.123 funziona")

    cfg["training_dataset_mean"] = 42.0
    assert cfg.training_dataset_mean == 42.0
    ok("campo Optional settato via dict interface")


# ─── 5. cfg.get(key, default) ────────────────────────────────────────────────

def test_get():
    print("\n[5/9] cfg.get(key, default)")
    cfg = NetworkConfig()

    assert cfg.get("train_step", 99) == 1       # campo esistente, ignora default
    assert cfg.get("WIDTH", 0) == 40
    assert cfg.get("chiave_mancante", 777) == 777  # campo inesistente → default
    ok("get() restituisce il valore del campo o il default")


# ─── 6. operatore in ─────────────────────────────────────────────────────────

def test_contains():
    print("\n[6/9] operatore in")
    cfg = NetworkConfig()

    assert "NUM_KC" in cfg
    assert "WIDTH" in cfg
    assert "name" in cfg
    assert "chiave_inesistente" not in cfg
    ok("'KEY' in cfg funziona correttamente")


# ─── 7. round-trip to_json / from_json ───────────────────────────────────────

def test_json_roundtrip():
    print("\n[7/9] round-trip to_json / from_json")
    original = NetworkConfig(NUM_KC=777, WIDTH=32, name="test_net", DT=0.5)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        original.to_json(path)
        loaded = NetworkConfig.from_json(path)

        assert loaded.NUM_KC == original.NUM_KC,   f"{loaded.NUM_KC} != {original.NUM_KC}"
        assert loaded.WIDTH == original.WIDTH,     f"{loaded.WIDTH} != {original.WIDTH}"
        assert loaded.name == original.name,       f"{loaded.name!r} != {original.name!r}"
        assert abs(loaded.DT - original.DT) < 1e-9
        ok("tutti i campi sono identici dopo to_json → from_json")

        # verifica che il file sia JSON valido e leggibile
        with open(path) as f:
            data = json.load(f)
        assert "NUM_KC" in data
        ok("il file prodotto è JSON valido con le chiavi attese")
    finally:
        os.unlink(path)


# ─── 8. chiavi sconosciute ignorate da from_json ─────────────────────────────

def test_unknown_keys_ignored():
    print("\n[8/9] chiavi sconosciute nel JSON ignorate da from_json")

    data = {
        "NUM_KC": 333,
        "WIDTH": 20,
        "chiave_non_esistente": "valore_qualsiasi",
        "altra_chiave_strana": 99999,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        cfg = NetworkConfig.from_json(path)
        assert cfg.NUM_KC == 333
        assert cfg.WIDTH == 20
        ok("chiavi sconosciute ignorate, campi noti caricati correttamente")
    finally:
        os.unlink(path)


# ─── 9. cfg.items() copre tutti i campi ──────────────────────────────────────

def test_items():
    print("\n[9/9] cfg.items() copre tutti i campi")
    cfg = NetworkConfig(NUM_KC=42)

    items_dict = dict(cfg.items())

    # tutte le field del dataclass devono essere presenti
    expected_keys = {f.name for f in fields(NetworkConfig)}
    assert expected_keys == set(items_dict.keys()), \
        f"chiavi mancanti: {expected_keys - set(items_dict.keys())}"

    assert items_dict["NUM_KC"] == 42
    ok(f"items() contiene tutti i {len(expected_keys)} campi del dataclass")


# ─── runner ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print(" Test NetworkConfig — insect_nav.config")
    print("=" * 50)

    tests = [
        test_defaults,
        test_kwargs,
        test_getitem,
        test_setitem,
        test_get,
        test_contains,
        test_json_roundtrip,
        test_unknown_keys_ignored,
        test_items,
    ]

    passed, failed_count = 0, 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  ✗  FALLITO: {e}")
            failed_count += 1

    print(f"\n{'='*50}")
    print(f" Risultato: {passed}/{len(tests)} test superati", end="")
    print(" ✓" if failed_count == 0 else f"  ({failed_count} falliti)")
    print("=" * 50)
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
