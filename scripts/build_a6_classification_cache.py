#!/usr/bin/env python3
"""Precompute the A6 Delhey colour classification for every specimen, in
every view (Back/Belly/Side) - protocole §3: classify_delhey_rgb() is
deterministic given the specimen - only the category *to remove* is drawn
per sac, not the classification itself.

Today that classification is redone independently by every consumer that
needs it: A6's own online draw (src/ablations/a6_colour_removal.py
draw_and_apply(), ~4-5s/specimen every time it's drawn) and the A6 presence
manifest (scripts/build_a6_presence_manifest.py, ~12s/specimen on the full
canvas). This script runs it once per specimen and caches the full per-pixel
label map, same "compute once offline instead of online" reasoning as A1/A7
(scripts/build_a1_cache.py, scripts/build_a7_cache.py).

Each cache entry is an int8 array, same shape as the source canvas's first
two dimensions: -1 for background, 0..11 indexing
src.ablations.a6_colour_removal.DELHEY_CATEGORIES for a foreground pixel.

configs/paths.yaml's dataset_dir/dataset_a6_classification_dir only name the
Back view (the one the protocol's own training/eval pipeline reads -
protocole §2, "vue Back uniquement"), but both the source dataset and this
cache actually hold all three views as sibling folders under their own
parent (NEW_Segmented-Aves-{view}-NPY, see scripts/build_grayscale_dataset.py)
- Familly_split_no_ablation for the source, Familly_split_A6_classification
for the cache. This script derives both parents from dataset_dir/
dataset_a6_classification_dir and, by default, builds all three views - the
Back one lands at exactly the path paths.yaml already names, Belly/Side are
siblings of it.

    python scripts/build_a6_classification_cache.py --workers 8
    python scripts/build_a6_classification_cache.py --views Belly Side   # skip an already-built Back
    python scripts/build_a6_classification_cache.py --limit 100   # smoke test, all views
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
from src.ablations import a6_colour_removal

VIEWS = ("Back", "Belly", "Side")
VIEW_TEMPLATE = "NEW_Segmented-Aves-{view}-NPY"

_DATASET_DIR = None
_DATASET_A6CLASS_DIR = None


def _init_worker(dataset_dir: str, dataset_a6class_dir: str) -> None:
    global _DATASET_DIR, _DATASET_A6CLASS_DIR
    _DATASET_DIR = Path(dataset_dir)
    _DATASET_A6CLASS_DIR = Path(dataset_a6class_dir)


def _process_one(stem: str) -> None:
    canvas = npy_io.read_canvas(_DATASET_DIR, stem)
    mask = npy_io.foreground_mask(canvas)
    labels = a6_colour_removal.build_classification(canvas, mask)
    a6_colour_removal.write_classification_cache_entry(_DATASET_A6CLASS_DIR, stem, labels)


def build_view(view: str, source_root: Path, dest_root: Path, workers: int, limit: Optional[int]) -> None:
    dataset_dir = source_root / VIEW_TEMPLATE.format(view=view)
    dataset_a6class_dir = dest_root / VIEW_TEMPLATE.format(view=view)
    dataset_a6class_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.name[: -len(".rgb.zst")] for p in dataset_dir.glob("*.rgb.zst"))
    if limit:
        stems = stems[:limit]

    already_done = {p.name[: -len(".delhey_labels.zst")] for p in dataset_a6class_dir.glob("*.delhey_labels.zst")}
    todo = [s for s in stems if s not in already_done]
    print(f"[{view}] {len(stems)} specimens total, {len(already_done)} already cached, {len(todo)} to do")
    if not todo:
        return

    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(str(dataset_dir), str(dataset_a6class_dir))
    ) as pool:
        list(tqdm(pool.map(_process_one, todo, chunksize=4), total=len(todo), desc=view))

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
    dataset_a6class_dir = Path(paths_cfg["dataset_a6_classification_dir"])
    source_root = _check_back_leaf_and_get_parent(dataset_dir, "dataset_dir")
    dest_root = _check_back_leaf_and_get_parent(dataset_a6class_dir, "dataset_a6_classification_dir")

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
