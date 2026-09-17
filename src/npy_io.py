"""Reading the *.rgb.zst / *.cand_128.zst / *.cand_224.zst cache files.

Each specimen in the dataset has three files, all the same container format:
a numpy array, np.save'd into an in-memory buffer, then compressed with zstd.
This module just knows how to undo that, plus how to write it back out (used
by scripts/build_a2_cache.py to produce the offline-standardized A2 dataset
in the same format so it can be read by the exact same loader).

Verified against a real specimen on disk: rgb.zst decompresses to a
(2024, 2024, 3) uint8 array, fond noir (background pixels are exactly 0 in
all three channels — the mask is reconstructed from that, per protocole §2).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import zstandard as zstd

_DECOMPRESSOR = zstd.ZstdDecompressor()


def read_array_zst(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        compressed = f.read()
    raw = _DECOMPRESSOR.decompress(compressed)
    return np.load(io.BytesIO(raw))


def write_array_zst(path: Path, array: np.ndarray, level: int = 12) -> None:
    buf = io.BytesIO()
    np.save(buf, array)
    compressed = zstd.ZstdCompressor(level=level).compress(buf.getvalue())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(compressed)


def read_canvas(dataset_dir: Path, stem: str) -> np.ndarray:
    """The (2024, 2024, 3) uint8 canvas for one specimen — 'lecture de la
    source', pipeline step 0."""
    return read_array_zst(Path(dataset_dir) / f"{stem}.rgb.zst")


def foreground_mask(canvas: np.ndarray) -> np.ndarray:
    """Pixels non nuls = plumage; fond noir = background (protocole §2, and
    the caveat in §2 about a genuinely black plumage pixel being miscounted
    as background applies here too)."""
    return canvas.any(axis=-1)
