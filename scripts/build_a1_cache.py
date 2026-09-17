#!/usr/bin/env python3
"""Precompute A1 (protocole §3: gris, L* conserve) for every specimen, in
every view (Back/Belly/Side).

A1 is a deterministic function of the specimen - no drawn parameter, no
condition-dependence - so recomputing it online for every sac that draws it
is pure waste: ~1.7s/specimen, every time, for the same result. This runs
it once per specimen and caches it, same shape as scripts/build_a2_cache.py.

configs/paths.yaml's dataset_dir/dataset_a1_dir only name the Back view (the
one the protocol's own training/eval pipeline reads - protocole §2, "vue
Back uniquement"), but both the source dataset and the A1 cache actually
hold all three views as sibling folders under their own parent
(NEW_Segmented-Aves-{view}-NPY, see scripts/build_grayscale_dataset.py) -
Familly_split_no_ablation for the source, Familly_split_A1 for the cache
(its own top-level folder, not nested inside Familly_split_no_ablation -
reorganised there on 2026-09-16). This script derives both parents from
dataset_dir/dataset_a1_dir and, by default, builds the A1 cache for all
three views - the Back one lands at exactly the path paths.yaml already
names, Belly/Side are siblings of it.

    python scripts/build_a1_cache.py --workers 8
    python scripts/build_a1_cache.py --views Belly Side   # skip the already-built Back
    python scripts/build_a1_cache.py --limit 100   # smoke test, all views
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import npy_io
from src.ablations import a1_gray_lstar

VIEWS = ("Back", "Belly", "Side")
VIEW_TEMPLATE = "NEW_Segmented-Aves-{view}-NPY"

_DATASET_DIR = None
_DATASET_A1_DIR = None


def _init_worker(dataset_dir: str, dataset_a1_dir: str) -> None:
    global _DATASET_DIR, _DATASET_A1_DIR
    _DATASET_DIR = Path(dataset_dir)
    _DATASET_A1_DIR = Path(dataset_a1_dir)


def _process_one(stem: str) -> None:
    canvas = npy_io.read_canvas(_DATASET_DIR, stem)
    mask = npy_io.foreground_mask(canvas)
    out_uint8 = a1_gray_lstar.build(canvas, mask)
    a1_gray_lstar.write_cache_entry(_DATASET_A1_DIR, stem, out_uint8)


def build_view(view: str, source_root: Path, dest_root: Path, workers: int, limit: Optional[int]) -> None:
    dataset_dir = source_root / VIEW_TEMPLATE.format(view=view)
    dataset_a1_dir = dest_root / VIEW_TEMPLATE.format(view=view)
    dataset_a1_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.name[: -len(".rgb.zst")] for p in dataset_dir.glob("*.rgb.zst"))
    if limit:
        stems = stems[:limit]

    already_done = {p.name[: -len(".rgb.zst")] for p in dataset_a1_dir.glob("*.rgb.zst")}
    todo = [s for s in stems if s not in already_done]
    print(f"[{view}] {len(stems)} specimens total, {len(already_done)} already cached, {len(todo)} to do")
    if not todo:
        return

    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(str(dataset_dir), str(dataset_a1_dir))
    ) as pool:
        list(tqdm(pool.map(_process_one, todo, chunksize=8), total=len(todo), desc=view))

    elapsed = time.time() - t0
    print(f"[{view}] done: {len(todo)} specimens in {elapsed:.0f}s ({elapsed/len(todo):.2f}s/specimen, {workers} workers)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"))
    parser.add_argument("--views", nargs="+", default=list(VIEWS), choices=VIEWS)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N specimens per view (smoke test)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with open(args.config) as f:
        paths_cfg = yaml.safe_load(f)
    dataset_dir = Path(paths_cfg["dataset_dir"])
    dataset_a1_dir = Path(paths_cfg["dataset_a1_dir"])
    source_root = _check_back_leaf_and_get_parent(dataset_dir, "dataset_dir")
    dest_root = _check_back_leaf_and_get_parent(dataset_a1_dir, "dataset_a1_dir")

    for view in args.views:
        source_dir = source_root / VIEW_TEMPLATE.format(view=view)
        if not source_dir.is_dir():
            raise FileNotFoundError(f"missing source view folder: {source_dir}")

    for view in args.views:
        build_view(view, source_root, dest_root, args.workers, args.limit)


def _check_back_leaf_and_get_parent(path: Path, config_key: str) -> Path:
    """`path` must be a 'NEW_Segmented-Aves-Back-NPY' folder directly under
    its parent - that parent is where the Belly/Side siblings are derived
    from. Raises loudly instead of silently building in the wrong place if
    configs/paths.yaml ever points this key somewhere else."""
    expected = path.parent / VIEW_TEMPLATE.format(view="Back")
    if path != expected:
        raise ValueError(
            f"configs/paths.yaml {config_key} ({path}) isn't a "
            f"'{VIEW_TEMPLATE.format(view='Back')}' folder directly under its parent - "
            f"can't derive the Belly/Side sibling folders from it."
        )
    return path.parent


if __name__ == "__main__":
    main()
