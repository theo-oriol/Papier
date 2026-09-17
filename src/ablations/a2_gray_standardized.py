"""A2 - gris + niveau et contraste standardises.

Standalone: no dependency on the rest of the project (only numpy +
zstandard). The CIELAB conversion is hand-rolled (sRGB <-> XYZ <-> Lab,
D65) instead of going through scikit-image, because — unlike A1/A6/A7 —
this file's ablation itself needs two passes over the *whole dataset*
before a single image can be processed, so it made sense to reuse the
already-verified standalone implementation from
scripts/build_grayscale_norm_lum_dataset.py rather than pull in skimage
for one extra dependency.

DEVIATION FROM THE PROTOCOL PDF, DELIBERATE: §3's A2 card specifies a
*fixed* target (L' = (L-m)*(15/s)+50, cible (50, 15)) - each specimen is
rescaled to its own fixed target, no dataset-wide computation involved.
This file no longer does that. Instead the target (mean M, spread S, and
the *full distribution*, not just its first two moments) is computed once
from the whole dataset and every specimen is matched to it - SHINE's
lumMatch + histMatch on L* (Willenbockel, Sadr, Fiset, Horne, Gosselin &
Tanaka, Behavior Research Methods 42:671-684, 2010), the same method
scripts/build_grayscale_norm_lum_dataset.py already uses for its own
(separate) grayscale variant. Decided explicitly over keeping A2 as
specified - see README "Ce qui n'a pas pu suivre le protocole a la
lettre". Three functions, matching three passes over the population:

  compute_stats()                - pass 1: this specimen's own (N, mean, std)
  compute_target_contribution()  - pass 2: its lum-matched histogram, to
                                     be accumulated (image-weighted) into
                                     one target density
  build()                         - pass 3: lumMatch + exact histMatch this
                                     specimen to the now-fixed (M, S, target)
                                     - the only one of the three that
                                     produces an output image

scripts/build_a2_cache.py orchestrates all three across every specimen and
caches (M, S, target_pdf) so they don't need recomputing on every rebuild.
Two jobs at the file-io level, same as A1/A7:
  write_cache_entry()/read_canvas() - the offline cache, read online by
                                        chain.py in place of the ordinary
                                        source read (pipeline step 0).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import zstandard as zstd

CODE = "A2"
N_BINS = 256
VMAX = 100.0  # L* range
WEIGHT = "image"  # one specimen = one vote, not one pixel = one vote (a large bird shouldn't dominate the target more than a small one)
CLIP_TARGET = True


# ---------------------------------------------------------------------------
# cache I/O
# ---------------------------------------------------------------------------

_DECOMPRESSOR = zstd.ZstdDecompressor()


def _read_array_zst(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        compressed = f.read()
    raw = _DECOMPRESSOR.decompress(compressed)
    return np.load(io.BytesIO(raw))


def _write_array_zst(path: Path, array: np.ndarray, level: int = 12) -> None:
    buf = io.BytesIO()
    np.save(buf, array)
    compressed = zstd.ZstdCompressor(level=level).compress(buf.getvalue())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(compressed)


def write_cache_entry(dataset_a2_dir: Path, stem: str, out_uint8: np.ndarray) -> None:
    _write_array_zst(Path(dataset_a2_dir) / f"{stem}.rgb.zst", out_uint8)


def read_canvas(dataset_a2_dir: Path, stem: str) -> np.ndarray:
    path = Path(dataset_a2_dir) / f"{stem}.rgb.zst"
    if not path.exists():
        raise FileNotFoundError(
            f"A2 cache entry missing: {path}\n"
            "Run scripts/build_a2_cache.py first (A2 needs a dataset-wide "
            "pass before any specimen can be standardised - see this "
            "module's docstring)."
        )
    return _read_array_zst(path)


# ---------------------------------------------------------------------------
# sRGB <-> CIELAB (D65) - same maths as build_grayscale_norm_lum_dataset.py
# ---------------------------------------------------------------------------

_M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)
_WHITE = np.array([0.95047, 1.0, 1.08883])
_DELTA = 6.0 / 29.0


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.maximum(c, 0) ** (1 / 2.4) - 0.055)


def _f(t: np.ndarray) -> np.ndarray:
    return np.where(t > _DELTA ** 3, np.cbrt(t), t / (3 * _DELTA ** 2) + 4.0 / 29.0)


def _f_inv(t: np.ndarray) -> np.ndarray:
    return np.where(t > _DELTA, t ** 3, 3 * _DELTA ** 2 * (t - 4.0 / 29.0))


def rgb_to_lab(rgb01: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1], shape (..., 3) -> CIELAB (L* in [0, 100])."""
    xyz = _srgb_to_linear(np.asarray(rgb01, dtype=np.float64)) @ _M_RGB2XYZ.T
    fx, fy, fz = [_f(xyz[..., i] / _WHITE[i]) for i in range(3)]
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIELAB -> sRGB in [0, 1], clipped."""
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    xyz = np.stack([_f_inv(fx) * _WHITE[0], _f_inv(fy) * _WHITE[1], _f_inv(fz) * _WHITE[2]], -1)
    return np.clip(_linear_to_srgb(xyz @ _M_XYZ2RGB.T), 0.0, 1.0)


# ---------------------------------------------------------------------------
# SHINE lumMatch / exact histMatch (Willenbockel et al. 2010, Table 1)
# ---------------------------------------------------------------------------

def lum_match(values: np.ndarray, M: float, S: float, vmax: float = VMAX, clip: bool = True) -> np.ndarray:
    """z-score then rescale onto (M, S), optionally clipped to [0, vmax]."""
    m, s = values.mean(), values.std()
    if s < 1e-8:
        return np.full_like(values, M)
    out = (values - m) / s * S + M
    return np.clip(out, 0.0, vmax) if clip else out


def _quantise(values: np.ndarray, n_bins: int, vmax: float) -> np.ndarray:
    step = vmax / (n_bins - 1)
    return np.clip(np.rint(values / step), 0, n_bins - 1).astype(np.int64)


def _counts_from_pdf(pdf: np.ndarray, n: int) -> np.ndarray:
    """Integer counts summing to exactly n, proportional to pdf (largest-
    remainder method): the target adapts to *this* image's own pixel count
    instead of assuming every image has the same N - the bug this whole
    approach exists to avoid (see bird_project's generate_meta_data_luminance.py)."""
    exact = np.asarray(pdf, dtype=np.float64) * n
    counts = np.floor(exact).astype(np.int64)
    short = n - int(counts.sum())
    if short > 0:
        order = np.argsort(-(exact - counts), kind="stable")
        counts[order[:short]] += 1
    elif short < 0:
        order = np.argsort(exact - counts, kind="stable")
        nz = order[counts[order] > 0][:-short]
        counts[nz] -= 1
    assert counts.sum() == n, (counts.sum(), n)
    return counts


def hist_match_exact(values: np.ndarray, pdf: np.ndarray, vmax: float = VMAX, rng=None) -> np.ndarray:
    """Exact histogram matching: pixels sorted ascending (ties broken at
    random), assigned the target's levels in order."""
    n = values.size
    counts = _counts_from_pdf(pdf, n)
    rng = np.random.default_rng() if rng is None else rng
    order = np.argsort(values + rng.uniform(0, 1e-9, size=n), kind="stable")
    levels = np.linspace(0.0, vmax, pdf.size)
    out = np.empty(n, dtype=np.float64)
    pos = 0
    for idx in np.nonzero(counts)[0]:
        c = int(counts[idx])
        out[order[pos:pos + c]] = levels[idx]
        pos += c
    assert pos == n, (pos, n)
    return out


# ---------------------------------------------------------------------------
# the three dataset-wide passes
# ---------------------------------------------------------------------------

def compute_stats(rgb_uint8: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Pass 1: this specimen's own L* (N, mean, std) - reduced across the
    whole dataset (image-weighted by default) into the population M, S."""
    lab = rgb_to_lab(rgb_uint8.astype(np.float64)[mask] / 255.0)
    values = lab[..., 0]
    return {"n_pixels": int(values.size), "mean": float(values.mean()), "std": float(values.std())}


def compute_target_contribution(
    rgb_uint8: np.ndarray, mask: np.ndarray, M: float, S: float, n_bins: int = N_BINS, clip_target: bool = CLIP_TARGET
) -> Dict[str, object]:
    """Pass 2: lum-match this specimen to (M, S), then histogram it - the
    weighted sum of every specimen's histogram (see build_a2_cache.py) is
    the target distribution pass 3 matches every specimen to."""
    lab = rgb_to_lab(rgb_uint8.astype(np.float64)[mask] / 255.0)
    values = lab[..., 0]

    raw = lum_match(values, M, S, VMAX, clip=False)
    frac_below_0 = float((raw < 0.0).mean())
    frac_above_max = float((raw > VMAX).mean())
    clipped = np.clip(raw, 0.0, VMAX)

    keep = np.ones(clipped.size, dtype=bool) if clip_target else (raw >= 0.0) & (raw <= VMAX)
    hist = np.zeros(n_bins, dtype=np.float64)
    if keep.sum() > 0:
        hist = np.bincount(_quantise(clipped[keep], n_bins, VMAX), minlength=n_bins) / keep.sum()

    return {
        "hist": hist,
        "frac_below_0": frac_below_0,
        "frac_above_max": frac_above_max,
        "mean_after_lum": float(clipped.mean()),
        "std_after_lum": float(clipped.std()),
    }


def build(
    rgb_uint8: np.ndarray,
    mask: np.ndarray,
    M: float,
    S: float,
    target_pdf: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Pass 3: lumMatch then exact histMatch this specimen's L* to the
    (already computed) population target. (a*, b*) untouched."""
    rgb01 = rgb_uint8.astype(np.float64) / 255.0
    lab = rgb_to_lab(rgb01)

    l_matched = hist_match_exact(lum_match(lab[..., 0][mask], M, S, VMAX), target_pdf, VMAX, rng)
    out_lab = lab.copy()
    out_lab[..., 0][mask] = l_matched

    rgb_out = lab_to_rgb(out_lab)
    out_uint8 = rgb_uint8.copy()
    out_uint8[mask] = np.rint(rgb_out[mask] * 255.0).astype(np.uint8)

    realised_lab = rgb_to_lab(out_uint8.astype(np.float64) / 255.0)
    realised_l = realised_lab[..., 0][mask]
    d_e = np.linalg.norm(realised_lab[mask] - out_lab[mask], axis=-1)
    meta = {
        "target_mean_L": M,
        "target_sd_L": S,
        "realized_mean_L": float(realised_l.mean()),
        "realized_sd_L": float(realised_l.std()),
        "dE_mean": float(d_e.mean()),
        "dE_max": float(d_e.max()),
    }
    return out_uint8, meta


def _demo_canvas(size: int = 256, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """A synthetic specimen for the __main__ example below: a few solid
    colour patches on a black background (background = pure zeros, the
    same "fond noir" convention as the real dataset; the mask is just "any
    channel nonzero")."""
    rng = np.random.default_rng(seed)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[40 : size - 40, 40 : size - 40] = rng.integers(
        60, 200, size=(size - 80, size - 80, 3), dtype=np.uint8
    )
    canvas[60:100, 60:140] = (200, 40, 40)  # a red patch
    canvas[110:150, 60:140] = (40, 60, 200)  # a blue patch
    canvas[160:200, 60:140] = (220, 200, 40)  # a yellow patch
    mask = canvas.any(axis=-1)
    return canvas, mask


if __name__ == "__main__":
    # A2 needs a *population*, not one image - five synthetic specimens
    # stand in for the dataset here.
    specimens = [_demo_canvas(seed=i) for i in range(5)]

    stats = [compute_stats(c, m) for c, m in specimens]
    w = np.ones(len(stats)) / len(stats)  # WEIGHT == "image"
    M = float(np.average([s["mean"] for s in stats], weights=w))
    S = float(np.average([s["std"] for s in stats], weights=w))
    print(f"A2: population M = {M:.2f}  S = {S:.2f}  (from {len(stats)} synthetic specimens)")

    contributions = [compute_target_contribution(c, m, M, S) for c, m in specimens]
    acc = sum(wi * c["hist"] for wi, c in zip(w, contributions))
    target_pdf = acc / acc.sum()
    print(f"target_pdf sums to 1: {np.isclose(target_pdf.sum(), 1.0)}")

    canvas, mask = specimens[0]
    out, meta = build(canvas, mask, M, S, target_pdf, rng=np.random.default_rng(0))
    print(f"specimen 0: input {canvas.shape} -> output {out.shape}")
    print(f"realized L* mean/sd: {meta['realized_mean_L']:.2f}/{meta['realized_sd_L']:.2f}  (target: {M:.2f}/{S:.2f})")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        write_cache_entry(tmp, "demo_specimen", out)
        reloaded = read_canvas(tmp, "demo_specimen")
        print("cache round-trip identical:", np.array_equal(out, reloaded))
