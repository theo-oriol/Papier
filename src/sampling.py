"""Position sampling for crops, from the precomputed candidate tables.

Protocole §2 step 5: 8 crops of 128px + 8 crops of 224px, 100% inside the
mask, positions separated by at least P/2 (P = crop side). Building this from
scratch per sac would mean scanning the whole 2024x2024 mask at 1px pitch
every time. Instead, the dataset ships *.cand_128.zst / *.cand_224.zst: every
admissible position (checked at 1px pitch, no grid) already computed once and
grouped into P/2-sized grid cells.

The candidate payload layout (confirmed on a real specimen file):
    payload[0]            -> kind (0 = no admissible window at all, 1 = only
                              one fallback position, 2 = the general case)
    payload[1]             -> n, the number of admissible positions
    payload[2 : 2+n]       -> y coordinates
    payload[2+n : 2+2n]    -> x coordinates
    payload[2+2n : 2+3n]   -> the P/2 grid-cell id each position falls into

Drawing at most one position per grid cell (shuffling the cell order first)
is how the "separation >= P/2" constraint from the protocol is actually
enforced: two positions in the same P/2 cell cannot both be kept.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .npy_io import read_array_zst


def load_candidates(dataset_dir: Path, stem: str, crop_size: int) -> dict:
    path = Path(dataset_dir) / f"{stem}.cand_{crop_size}.zst"
    payload = read_array_zst(path)
    return _parse_candidates(payload)


def _parse_candidates(payload: np.ndarray) -> dict:
    kind = int(payload[0])
    if kind == 0:
        return {"edge_case": True}
    if kind == 1:
        return {"fallback": (int(payload[1]), int(payload[2]))}

    n = int(payload[1])
    valid_ys = payload[2 : 2 + n]
    valid_xs = payload[2 + n : 2 + 2 * n]
    cell_id = payload[2 + 2 * n : 2 + 3 * n]
    unique_cells = np.unique(cell_id)
    in_cell = {int(c): np.where(cell_id == c)[0] for c in unique_cells}
    return {
        "valid_ys": valid_ys,
        "valid_xs": valid_xs,
        "unique_cells": unique_cells,
        "in_cell": in_cell,
    }


def sample_positions(candidates: dict, n_positions: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    """Draw n_positions (y, x) top-left corners, one grid cell at a time so
    consecutive draws stay >= P/2 apart, per the sac's own RNG."""
    if candidates.get("edge_case"):
        return [(0, 0)] * n_positions
    if "fallback" in candidates:
        return [candidates["fallback"]] * n_positions

    valid_ys = candidates["valid_ys"]
    valid_xs = candidates["valid_xs"]
    cells = candidates["unique_cells"].copy()
    rng.shuffle(cells)

    positions: List[Tuple[int, int]] = []
    for cell in cells:
        if len(positions) >= n_positions:
            break
        choices = candidates["in_cell"][int(cell)]
        chosen = choices[rng.integers(len(choices))]
        positions.append((int(valid_ys[chosen]), int(valid_xs[chosen])))

    # more crops requested than distinct P/2 cells exist (small specimens):
    # fall back to plain uniform draws for the remainder.
    while len(positions) < n_positions:
        idx = rng.integers(len(valid_ys))
        positions.append((int(valid_ys[idx]), int(valid_xs[idx])))

    return positions
