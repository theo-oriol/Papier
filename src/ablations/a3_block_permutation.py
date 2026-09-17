"""A3 - permutation de blocs (protocole §3 et l'encadre "A3 - blocs carres").

Cuts the 224px crop into square h x h blocks and permutes them (a strict
permutation: every block moves, none is rotated or resampled). Removes
large-scale spatial layout while leaving local texture statistics (and the
crop's histogram) untouched.

Applied to the crop itself, *never* the full image (permuting the whole
bird would move plumage between body regions, and would be a no-op for most
crops anyway - see protocole §3 "Deux choix de mise en oeuvre"). Drawn per
crop, not per sac: each of the 16 crops in a sac gets its own h and its own
arrangement, so a sac under A3 is a mean over several severities, not one.

h is drawn *uniformly* over the ten divisors of 224 that are <= 56 (not
weighted 1/h - the previous, non-uniform version of this protocol put 44.7%
of the mass on h=1, see protocole §3 "A3 - blocs carres").
"""

from __future__ import annotations

import numpy as np

CODE = "A3"
CROP_SIDE = 224
SIDE_CHOICES = (1, 2, 4, 7, 8, 14, 16, 28, 32, 56)


def draw_side(rng: np.random.Generator) -> int:
    return int(SIDE_CHOICES[rng.integers(len(SIDE_CHOICES))])


def apply(crop_uint8: np.ndarray, h: int, rng: np.random.Generator) -> np.ndarray:
    n = CROP_SIDE // h
    blocks = crop_uint8.reshape(n, h, n, h, 3).transpose(0, 2, 1, 3, 4).reshape(n * n, h, h, 3)
    order = rng.permutation(n * n)
    permuted = blocks[order].reshape(n, n, h, h, 3).transpose(0, 2, 1, 3, 4).reshape(CROP_SIDE, CROP_SIDE, 3)
    return permuted


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 256, size=(CROP_SIDE, CROP_SIDE, 3), dtype=np.uint8)

    h = draw_side(rng)
    permuted = apply(crop, h, rng)

    print(f"A3: side drawn = {h}px, crop {crop.shape} -> {permuted.shape}")
    print(
        "histogram preserved (same pixel values, just rearranged):",
        np.array_equal(np.sort(crop.ravel()), np.sort(permuted.ravel())),
    )
