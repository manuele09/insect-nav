"""
Test delle metriche di novelty di insect_nav.

Cosa viene testato, in ordine:
    1. references vuota      — tutti i valori sono 0
    2. confronto con se stesso — cosine≈0, pearson≈0, euclidean≈0
    3. vettori diversi        — tutti i valori > 0
    4. minimo sui riferimenti — novelty(A vs [A,B]) ≤ novelty(A vs [B])
    5. chiavi del dizionario  — esattamente {"cosine", "pearson", "euclidean"}
    6. tipi restituiti        — tutti float Python, non numpy scalars

Esecuzione:
    python tests/test_metrics.py
    pytest tests/test_metrics.py -v
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from insect_nav.metrics import novelty_scores

# ─── vettori sintetici di test ────────────────────────────────────────────────

rng = np.random.default_rng(0)

VEC_A = rng.random(48).astype(np.float32)          # vettore di riferimento
VEC_B = rng.random(48).astype(np.float32)          # vettore diverso
VEC_A_NOISY = VEC_A + rng.random(48) * 0.01        # quasi identico ad A


def ok(msg):
    print(f"  ✓  {msg}")

def fail(msg):
    print(f"  ✗  {msg}")
    raise AssertionError(msg)


# ─── 1. references vuota ─────────────────────────────────────────────────────

def test_empty_references():
    print("\n[1/6] references vuota")
    scores = novelty_scores(VEC_A, [])
    for metric, value in scores.items():
        assert value == 0, f"{metric} atteso 0, ottenuto {value}"
    ok("tutti i valori sono 0 quando references=[]")


# ─── 2. confronto con se stesso ──────────────────────────────────────────────

def test_self_comparison():
    print("\n[2/6] confronto con se stesso")
    scores = novelty_scores(VEC_A, [VEC_A])
    assert scores["cosine"] < 1e-6, \
        f"cosine atteso ≈0, ottenuto {scores['cosine']}"
    assert scores["pearson"] < 1e-6, \
        f"pearson atteso ≈0, ottenuto {scores['pearson']}"
    assert scores["euclidean"] < 1e-6, \
        f"euclidean atteso ≈0, ottenuto {scores['euclidean']}"
    ok(f"cosine={scores['cosine']:.2e}, pearson={scores['pearson']:.2e}, "
       f"euclidean={scores['euclidean']:.2e}")


# ─── 3. vettori diversi ──────────────────────────────────────────────────────

def test_different_vectors():
    print("\n[3/6] vettori diversi")
    scores = novelty_scores(VEC_A, [VEC_B])
    assert scores["cosine"] > 0, f"cosine atteso > 0, ottenuto {scores['cosine']}"
    assert scores["pearson"] > 0, f"pearson atteso > 0, ottenuto {scores['pearson']}"
    assert scores["euclidean"] > 0, f"euclidean atteso > 0, ottenuto {scores['euclidean']}"
    ok(f"cosine={scores['cosine']:.4f}, pearson={scores['pearson']:.4f}, "
       f"euclidean={scores['euclidean']:.4f}")


# ─── 4. minimo sui riferimenti ───────────────────────────────────────────────

def test_minimum_over_references():
    print("\n[4/6] minimo sui riferimenti")
    # novelty(A vs [A_noisy, B]) deve essere ≤ novelty(A vs [B])
    # perché A_noisy è molto simile ad A
    scores_b_only = novelty_scores(VEC_A, [VEC_B])
    scores_ab = novelty_scores(VEC_A, [VEC_A_NOISY, VEC_B])

    for metric in ("cosine", "pearson", "euclidean"):
        assert scores_ab[metric] <= scores_b_only[metric], (
            f"{metric}: novelty vs [A_noisy,B]={scores_ab[metric]:.4f} "
            f"dovrebbe essere ≤ novelty vs [B]={scores_b_only[metric]:.4f}"
        )
    ok("novelty(A vs [A_noisy,B]) ≤ novelty(A vs [B]) per tutte le metriche")


# ─── 5. chiavi del dizionario ────────────────────────────────────────────────

def test_dict_keys():
    print("\n[5/6] chiavi del dizionario")
    scores = novelty_scores(VEC_A, [VEC_B])
    expected_keys = {"cosine", "pearson", "euclidean"}
    assert set(scores.keys()) == expected_keys, \
        f"chiavi attese {expected_keys}, ottenute {set(scores.keys())}"
    ok(f"chiavi corrette: {sorted(scores.keys())}")


# ─── 6. tipi restituiti ──────────────────────────────────────────────────────

def test_return_types():
    print("\n[6/6] tipi restituiti")
    scores = novelty_scores(VEC_A, [VEC_B])
    for metric, value in scores.items():
        assert isinstance(value, float), \
            f"{metric}: atteso float, ottenuto {type(value).__name__}"
    ok("tutti i valori sono float Python")


# ─── runner ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print(" Test metriche di novelty — insect_nav.metrics")
    print("=" * 50)

    tests = [
        test_empty_references,
        test_self_comparison,
        test_different_vectors,
        test_minimum_over_references,
        test_dict_keys,
        test_return_types,
    ]

    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            fail(str(e))
            failed += 1

    print(f"\n{'='*50}")
    print(f" Risultato: {passed}/{len(tests)} test superati", end="")
    print(" ✓" if failed == 0 else f"  ({failed} falliti)")
    print("=" * 50)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
