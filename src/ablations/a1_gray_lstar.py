"""A1 - gris, L* conserve (protocole §3).

Standalone: this file has no dependency on the rest of the project (only
numpy + scikit-image + zstandard, all third-party) - copy this one file
anywhere and it works on its own. The CIELAB conversion helpers below are
copied verbatim from src/color.py (the project's shared colour module)
rather than imported from it, specifically so this file doesn't need
color.py sitting next to it to run.

Removes chromaticity pixel by pixel in CIELAB (a* = b* = 0) while keeping
L* exactly, on the whole bird ("oiseau entier"). No drawn parameter -
deterministic given the specimen alone, so it is precomputed once offline
(scripts/build_a1_cache.py) instead of being recomputed by every sac that
draws it online across every epoch.

Two jobs:
  build()       - run once, offline, by scripts/build_a1_cache.py.
  read_canvas() - run online, by chain.py, in place of the ordinary
                   source read (this ablation replaces pipeline step 0,
                   not step 1 - see src/ablations/common.py).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import zstandard as zstd
from skimage import color


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


def _check_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected H x W x 3 RGB array, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("RGB image contains non-finite values")
    if arr.min() < -1e-8 or arr.max() > 1.0 + 1e-8:
        raise ValueError("RGB input must be scaled to [0, 1]")
    return np.clip(arr, 0.0, 1.0)


def _check_foreground(mask: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"Foreground mask shape {mask.shape} != image shape {shape}")
    if not np.any(mask):
        raise ValueError("Foreground mask is empty")
    return mask


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an sRGB image in [0, 1] to CIE Lab under D65."""

    return color.rgb2lab(_check_rgb(rgb), illuminant="D65")


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIE Lab under D65 to clipped sRGB in [0, 1]."""

    lab = np.asarray(lab, dtype=np.float64)
    if lab.ndim != 3 or lab.shape[-1] != 3:
        raise ValueError(f"Expected H x W x 3 Lab array, got {lab.shape}")
    with np.errstate(invalid="ignore"):
        rgb = color.lab2rgb(lab, illuminant="D65")
    return np.clip(rgb, 0.0, 1.0)


def achromatic_raw(
    rgb: np.ndarray, foreground: Optional[np.ndarray] = None
) -> np.ndarray:
    """Remove all chroma while retaining each foreground pixel's L*."""

    arr = _check_rgb(rgb)
    lab = rgb_to_lab(arr)
    fg = _check_foreground(foreground, lab.shape[:2])
    lab[..., 1][fg] = 0.0
    lab[..., 2][fg] = 0.0
    result = lab_to_rgb(lab)
    result[~fg] = arr[~fg]
    return result


CODE = "A1"


def build(rgb_uint8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb01 = rgb_uint8.astype(np.float64) / 255.0
    out01 = achromatic_raw(rgb01, foreground=mask)
    return np.clip(out01 * 255.0, 0, 255).astype(np.uint8)


def write_cache_entry(dataset_a1_dir: Path, stem: str, out_uint8: np.ndarray) -> None:
    _write_array_zst(Path(dataset_a1_dir) / f"{stem}.rgb.zst", out_uint8)


def read_canvas(dataset_a1_dir: Path, stem: str) -> np.ndarray:
    path = Path(dataset_a1_dir) / f"{stem}.rgb.zst"
    if not path.exists():
        raise FileNotFoundError(
            f"A1 cache entry missing: {path}\n"
            "Run scripts/build_a1_cache.py first (A1 is deterministic "
            "given the specimen, so it's precomputed once instead of online)."
        )
    return _read_array_zst(path)


def _demo_canvas(size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """A synthetic specimen for the __main__ example below: a few solid
    colour patches on a black background (background = pure zeros, the
    same "fond noir" convention as the real dataset; the mask is just "any
    channel nonzero")."""
    rng = np.random.default_rng(0)
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
    import tempfile

    canvas, mask = _demo_canvas()
    out = build(canvas, mask)

    spread_before = int(np.ptp(canvas[mask].astype(int), axis=1).max())
    spread_after = int(np.ptp(out[mask].astype(int), axis=1).max())
    print(f"A1: input {canvas.shape} -> output {out.shape}")
    print(
        f"max per-pixel R/G/B spread before: {spread_before}  after: {spread_after} "
        "(should be ~0, image is now achromatic)"
    )

    with tempfile.TemporaryDirectory() as tmp:
        write_cache_entry(tmp, "demo_specimen", out)
        reloaded = read_canvas(tmp, "demo_specimen")
        print("cache round-trip identical:", np.array_equal(out, reloaded))
