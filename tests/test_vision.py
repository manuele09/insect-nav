"""
Test della pipeline visuale di insect_nav.

Cosa viene testato, in ordine:
    1. cropFrame       — ritaglio top/bottom
    2. shiftFrame      — shift orizzontale con wrap-around
    3. preprocessFrame — pipeline completa: crop → grayscale+invert → shift → resize
    4. extractFeatures — distribuzione verticale e/o orizzontale

Esecuzione:
    python tests/test_vision.py
    pytest tests/test_vision.py -v
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from insect_nav.vision import (
    cropFrame,
    shiftFrame,
    preprocessFrame,
    extractFeatures,
)
from insect_nav.config import NetworkConfig

# ─── immagine sintetica di test ───────────────────────────────────────────────
# Gradiente orizzontale BGR: utile perché il comportamento di shift e crop
# è prevedibile analiticamente (le colonne hanno valori crescenti).

H_IN, W_IN = 240, 320

def make_gradient_bgr(h=H_IN, w=W_IN) -> np.ndarray:
    """Immagine BGR con gradiente orizzontale da 0 a 255."""
    row = np.linspace(0, 255, w, dtype=np.uint8)
    frame = np.stack([np.tile(row, (h, 1))] * 3, axis=-1)  # (H, W, 3)
    return frame


def make_random_bgr(seed=42, h=H_IN, w=W_IN) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ─── parametri di configurazione minimi ──────────────────────────────────────

BASE_CFG = dict(
    WIDTH=40, HEIGHT=8,
    CROP_TOP=20, CROP_BOTTOM=20,
    USE_VERTICAL_DIST=True,
    USE_HORIZONTAL_DIST=False,
    VERTICAL_WEIGHT=0.0,
    HORIZONTAL_WEIGHT=0.0,
)


def ok(msg):
    print(f"  ✓  {msg}")

def fail(msg):
    print(f"  ✗  {msg}")
    raise AssertionError(msg)


# ─── 1. cropFrame ─────────────────────────────────────────────────────────────

def test_cropFrame():
    print("\n[1/4] cropFrame")
    frame = make_gradient_bgr()

    # a) dimensioni corrette dopo il crop
    crop_top, crop_bottom = 20, 40
    cropped = cropFrame(frame, crop_bottom=crop_bottom, crop_top=crop_top)
    expected_h = H_IN - crop_top - crop_bottom
    assert cropped.shape == (expected_h, W_IN, 3), \
        f"shape attesa {(expected_h, W_IN, 3)}, ottenuta {cropped.shape}"
    ok(f"shape dopo crop ({crop_top}px top, {crop_bottom}px bottom): {cropped.shape}")

    # b) crop=0 lascia l'immagine invariata
    unchanged = cropFrame(frame, crop_bottom=0, crop_top=0)
    assert np.array_equal(unchanged, frame)
    ok("crop=0 → immagine identica")

    # c) il contenuto rimasto è quello corretto (non le righe rimosse)
    assert np.array_equal(cropped, frame[crop_top: H_IN - crop_bottom, :, :])
    ok("i pixel conservati coincidono con la slice attesa")


# ─── 2. shiftFrame ────────────────────────────────────────────────────────────

def test_shiftFrame():
    print("\n[2/4] shiftFrame")
    # shiftFrame lavora su immagini float2D (output di cropFrame+cvtColor)
    img = np.linspace(0, 255, W_IN, dtype=np.float32)
    img = np.tile(img, (H_IN, 1))   # gradiente uniforme per riga

    # a) shift 0° → identica
    shifted_0 = shiftFrame(img, 0)
    assert np.array_equal(shifted_0, img)
    ok("shift 0° → immagine identica")

    # b) shift 360° → wrap completo → identica
    shifted_360 = shiftFrame(img, 360)
    assert np.allclose(shifted_360, img, atol=1.0)   # atol per arrotondamento pixel
    ok("shift 360° → wrap completo → identica (atol=1)")

    # c) shift positivo → i valori si spostano a destra
    # Il pixel in posizione 0 dopo uno shift di 90° deve essere uguale
    # al pixel che era a -90°/360° * W_IN colonne dall'inizio
    shifted_90 = shiftFrame(img, 90)
    x_offset = int((90 / 360.0) * W_IN)       # colonne spostate
    assert shifted_90[0, x_offset] == img[0, 0], \
        f"valore atteso {img[0,0]:.1f}, ottenuto {shifted_90[0, x_offset]:.1f}"
    ok(f"shift 90° → colonna 0 si trova a offset={x_offset}")

    # d) shift negativo → opposto dello shift positivo
    shifted_pos = shiftFrame(img, 45)
    shifted_neg = shiftFrame(img, -315)    # -315 ≡ +45 mod 360
    assert np.allclose(shifted_pos, shifted_neg, atol=1.0)
    ok("shift +45° ≡ shift -315° (stessa posizione mod 360)")


# ─── 3. preprocessFrame ───────────────────────────────────────────────────────

def test_preprocessFrame():
    print("\n[3/4] preprocessFrame")
    cfg = NetworkConfig(**BASE_CFG)
    frame = make_gradient_bgr()

    # a) output ha le dimensioni corrette (HEIGHT x WIDTH)
    out = preprocessFrame(frame, shift_degrees=0, parameters_dict=cfg)
    assert out.shape == (cfg.HEIGHT, cfg.WIDTH), \
        f"shape attesa ({cfg.HEIGHT}, {cfg.WIDTH}), ottenuta {out.shape}"
    ok(f"output shape: {out.shape}  (HEIGHT={cfg.HEIGHT}, WIDTH={cfg.WIDTH})")

    # b) i valori sono invertiti: l'immagine originale era un gradiente
    # chiaro→scuro, dopo l'inversione è scuro→chiaro
    assert out.dtype == np.float32
    ok(f"dtype float32 corretto")

    # c) shift 0° e shift 360° producono lo stesso output
    out_0 = preprocessFrame(frame, shift_degrees=0, parameters_dict=cfg)
    out_360 = preprocessFrame(frame, shift_degrees=360, parameters_dict=cfg)
    assert np.allclose(out_0, out_360, atol=1.0)
    ok("preprocessFrame(shift=0°) ≈ preprocessFrame(shift=360°)")

    # d) shift diversi producono output diversi
    out_90 = preprocessFrame(frame, shift_degrees=90, parameters_dict=cfg)
    assert not np.allclose(out_0, out_90)
    ok("shift 0° e shift 90° producono output diversi")


# ─── 4. extractFeatures ───────────────────────────────────────────────────────

def test_extractFeatures():
    print("\n[4/4] extractFeatures")
    cfg_v  = NetworkConfig(**{**BASE_CFG, "USE_VERTICAL_DIST": True,  "USE_HORIZONTAL_DIST": False})
    cfg_h  = NetworkConfig(**{**BASE_CFG, "USE_VERTICAL_DIST": False, "USE_HORIZONTAL_DIST": True})
    cfg_vh = NetworkConfig(**{**BASE_CFG, "USE_VERTICAL_DIST": True,  "USE_HORIZONTAL_DIST": True})

    frame = make_gradient_bgr()
    preprocessed = preprocessFrame(frame, 0, cfg_v)   # (HEIGHT, WIDTH) float32

    # a) solo distribuzione verticale → lunghezza = WIDTH
    feats_v = extractFeatures(preprocessed, cfg_v)
    assert feats_v.shape == (cfg_v.WIDTH,), \
        f"atteso ({cfg_v.WIDTH},), ottenuto {feats_v.shape}"
    ok(f"USE_VERTICAL_DIST  → len={len(feats_v)} (== WIDTH={cfg_v.WIDTH})")

    # b) solo distribuzione orizzontale → lunghezza = HEIGHT
    feats_h = extractFeatures(preprocessed, cfg_h)
    assert feats_h.shape == (cfg_h.HEIGHT,), \
        f"atteso ({cfg_h.HEIGHT},), ottenuto {feats_h.shape}"
    ok(f"USE_HORIZONTAL_DIST → len={len(feats_h)} (== HEIGHT={cfg_h.HEIGHT})")

    # c) entrambe → lunghezza = WIDTH + HEIGHT
    feats_vh = extractFeatures(preprocessed, cfg_vh)
    assert feats_vh.shape == (cfg_vh.WIDTH + cfg_vh.HEIGHT,), \
        f"atteso ({cfg_vh.WIDTH + cfg_vh.HEIGHT},), ottenuto {feats_vh.shape}"
    ok(f"entrambe le distribuzioni  → len={len(feats_vh)} (== WIDTH+HEIGHT={cfg_vh.WIDTH + cfg_vh.HEIGHT})")

    # d) il vettore non è costante (contiene informazione)
    assert feats_v.std() > 0
    ok("il vettore di feature non è costante (std > 0)")



# ─── runner ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print(" Test pipeline visuale — insect_nav.vision")
    print("=" * 50)

    tests = [
        test_cropFrame,
        test_shiftFrame,
        test_preprocessFrame,
        test_extractFeatures,
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
