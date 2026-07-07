"""
Test di insect_nav.variants — generazione varianti di parametri (nessuna rete
neurale reale coinvolta: copertura di generate_parameter_variants e
apply_parameter_transformation con network_type="infomax"/"perfect_memory",
che non toccano NeuralNetwork/pygenn).

Esecuzione:
    python tests/test_variants.py
    pytest tests/test_variants.py -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from insect_nav.variants import apply_parameter_transformation, generate_parameter_variants


def ok(msg):
    print(f"  ok  {msg}")

def fail(msg):
    print(f"  FAIL  {msg}")
    raise AssertionError(msg)


BASE_PARAMS = {
    "network_type": "infomax",
    "name": "base_net",
    "learning_rate": 0.1,
    "output_units": 64,
    "train_step": 2,
    "IF_PARAMS": {"Vthresh": 30},
    "trainingDatasetPath": "/home/placeholder/Dataset/fake_dataset",
}


# ─── generate_parameter_variants ──────────────────────────────────────────────

def test_no_transformations_returns_single_copy():
    print("\n[1/5] nessuna trasformazione -> singola copia di base_params")
    variants = generate_parameter_variants(BASE_PARAMS, {})
    assert len(variants) == 1
    assert variants[0] == BASE_PARAMS
    ok("ritorna esattamente una copia identica")


def test_cartesian_product():
    print("\n[2/5] prodotto cartesiano su piu' parametri")
    variants = generate_parameter_variants(
        BASE_PARAMS,
        {"train_step": [2, 4], "learning_rate": [0.1, 0.2, 0.3]},
    )
    assert len(variants) == 6, f"attese 6 combinazioni, ottenute {len(variants)}"
    combos = {(v["train_step"], v["learning_rate"]) for v in variants}
    expected = {(ts, lr) for ts in (2, 4) for lr in (0.1, 0.2, 0.3)}
    assert combos == expected
    ok("6 combinazioni, tutte le coppie (train_step, learning_rate) attese presenti")


def test_dotted_key_nested():
    print("\n[3/5] chiave puntata per dict annidato (IF_PARAMS.Vthresh)")
    variants = generate_parameter_variants(BASE_PARAMS, {"IF_PARAMS.Vthresh": [10, 20]})
    assert len(variants) == 2
    assert {v["IF_PARAMS"]["Vthresh"] for v in variants} == {10, 20}
    ok("IF_PARAMS.Vthresh impostato correttamente su ciascuna variante")


def test_no_shared_state_between_variants():
    print("\n[4/5] deep copy: nessun leak di stato tra varianti (dict annidati non condivisi)")
    variants = generate_parameter_variants(BASE_PARAMS, {"IF_PARAMS.Vthresh": [10, 20, 30]})
    assert variants[0]["IF_PARAMS"] is not variants[1]["IF_PARAMS"]
    assert variants[0]["IF_PARAMS"]["Vthresh"] == 10
    assert variants[1]["IF_PARAMS"]["Vthresh"] == 20
    assert variants[2]["IF_PARAMS"]["Vthresh"] == 30
    assert BASE_PARAMS["IF_PARAMS"]["Vthresh"] == 30, "base_params non deve essere mutato"
    ok("dict annidati indipendenti tra varianti, base_params non mutato")


# ─── apply_parameter_transformation ───────────────────────────────────────────

def test_apply_parameter_transformation_infomax():
    print("\n[5/5] apply_parameter_transformation su network_type=infomax (no NeuralNetwork/pygenn)")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source_dir = tmp / "source_net"
        source_dir.mkdir()
        source_params_path = source_dir / "parameters.json"
        params = dict(BASE_PARAMS)
        with open(source_params_path, "w") as f:
            json.dump(params, f)

        output_dir = tmp / "variants"
        created = apply_parameter_transformation(
            source_params_path=str(source_params_path),
            transformations={"train_step": [2, 4]},
            output_base_dir=str(output_dir),
            copy_weights=False,
        )

        assert len(created) == 2
        for p in created:
            assert Path(p).is_file(), f"parameters.json mancante: {p}"
            with open(p) as f:
                saved = json.load(f)
            assert saved["train_step"] in (2, 4)
            assert Path(saved["plotsTrainPath"]).is_dir()
            assert Path(saved["plotsTestPath"]).is_dir()
            assert Path(saved["plotsSimulationPath"]).is_dir()
        ok("2 varianti create con parameters.json + directory dei plot")


# ─── runner ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print(" Test insect_nav.variants")
    print("=" * 50)

    tests = [
        test_no_transformations_returns_single_copy,
        test_cartesian_product,
        test_dotted_key_nested,
        test_no_shared_state_between_variants,
        test_apply_parameter_transformation_infomax,
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
