#!/usr/bin/env python3
"""Precompute A7 (protocole §3: chromaticite seule, L* fixe) for every
specimen, in every view (Back/Belly/Side).

A7 is a deterministic function of the specimen, and it was the single
slowest ablation measured (~7-8s/specimen - color.chroma_isoluminant()
classifies the whole bird internally just to report gamut-clipping
statistics per Delhey category). Recomputing that online for every sac
that draws A7 is pure waste, so - same as A1/A2 - this runs it once per
specimen and caches it.

Alongside the ablated RGB, each entry also caches the specimen's full-canvas
*input* Lab (float32, one {stem}.lab.zst file per {stem}.rgb.zst) -
chroma_isoluminant()'s own rgb_to_lab() conversion, the most expensive step
in it on a 2024x2024 canvas, paid for here and reusable afterwards instead
of being redone by anything else that needs this same specimen's Lab
(A1's achromatic_raw(), A6's classify_delhey_rgb()).

Also writes runs/a7_gamut_metadata/<view>.csv (one row per specimen): the
gamut retention / delta-E statistics that used to live in the per-sac
metadata before A7 became a cache read. protocole §3 is explicit that A7
must not be described as exactly chroma-preserving without that number
attached - this is where it now lives, computed once instead of never.

configs/paths.yaml's dataset_dir/dataset_a7_dir only name the Back view (the
one the protocol's own training/eval pipeline reads - protocole §2, "vue
Back uniquement"), but both the source dataset and the A7 cache actually
hold all three views as sibling folders under their own parent
(NEW_Segmented-Aves-{view}-NPY, see scripts/build_grayscale_dataset.py) -
Familly_split_no_ablation for the source, Familly_split_A7 for the cache
(its own top-level folder, not nested inside Familly_split_no_ablation -
reorganised there on 2026-09-16). This script derives both parents from
dataset_dir/dataset_a7_dir and, by default, builds the A7 cache for all
three views - the Back one lands at exactly the path paths.yaml already
names, Belly/Side are siblings of it.

    python scripts/build_a7_cache.py --workers 8
    python scripts/build_a7_cache.py --views Belly Side   # skip an already-built Back
    python scripts/build_a7_cache.py --limit 100   # smoke test, all views
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import npy_io
from src.ablations import a7_chroma_only

VIEWS = ("Back", "Belly", "Side")
VIEW_TEMPLATE = "NEW_Segmented-Aves-{view}-NPY"

GAMUT_METADATA_COLUMNS = [
    "stem",
    "mean_intended_chroma",
    "mean_realized_chroma",
    "mean_chroma_retention_ratio",
    "median_chroma_retention_ratio",
    "fraction_chromatic_pixels_losing_gt_5pct_chroma",
    "fraction_chromatic_pixels_losing_gt_20pct_chroma",
    "mean_delta_E_roundtrip",
    "fraction_pixels_delta_E_gt_1",
    "gamut_or_roundtrip_distortion_detected",
]

_DATASET_DIR = None
_DATASET_A7_DIR = None


def _init_worker(dataset_dir: str, dataset_a7_dir: str) -> None:
    global _DATASET_DIR, _DATASET_A7_DIR
    _DATASET_DIR = Path(dataset_dir)
    _DATASET_A7_DIR = Path(dataset_a7_dir)


def _process_one(stem: str) -> dict:
    canvas = npy_io.read_canvas(_DATASET_DIR, stem)
    mask = npy_io.foreground_mask(canvas)
    out_uint8, lab_float32, meta = a7_chroma_only.build(canvas, mask)
    a7_chroma_only.write_cache_entry(_DATASET_A7_DIR, stem, out_uint8)
    a7_chroma_only.write_lab_cache_entry(_DATASET_A7_DIR, stem, lab_float32)
    row = {"stem": stem}
    row.update({key: meta[key] for key in GAMUT_METADATA_COLUMNS if key != "stem"})
    return row


def build_view(view: str, source_root: Path, dest_root: Path, metadata_root: Path, workers: int, limit: Optional[int]) -> None:
    dataset_dir = source_root / VIEW_TEMPLATE.format(view=view)
    dataset_a7_dir = dest_root / VIEW_TEMPLATE.format(view=view)
    dataset_a7_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = metadata_root / f"{view}.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.name[: -len(".rgb.zst")] for p in dataset_dir.glob("*.rgb.zst"))
    if limit:
        stems = stems[:limit]

    # A stem only counts as done once *both* its RGB and Lab entries exist -
    # a run killed between the two writes for one specimen must redo that
    # specimen rather than being left with a Lab-less (or RGB-less) entry.
    has_rgb = {p.name[: -len(".rgb.zst")] for p in dataset_a7_dir.glob("*.rgb.zst")}
    has_lab = {p.name[: -len(".lab.zst")] for p in dataset_a7_dir.glob("*.lab.zst")}
    already_done = has_rgb & has_lab
    todo = [s for s in stems if s not in already_done]
    print(f"[{view}] {len(stems)} specimens total, {len(already_done)} already cached, {len(todo)} to do")
    if not todo:
        return

    write_header = not metadata_path.exists()
    t0 = time.time()
    with open(metadata_path, "a", newline="") as f, ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(str(dataset_dir), str(dataset_a7_dir))
    ) as pool:
        writer = csv.DictWriter(f, fieldnames=GAMUT_METADATA_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in tqdm(pool.map(_process_one, todo, chunksize=8), total=len(todo), desc=view):
            writer.writerow(row)
            f.flush()  # every row committed to disk - a killed run loses nothing

    elapsed = time.time() - t0
    print(f"[{view}] done: {len(todo)} specimens in {elapsed:.0f}s ({elapsed/len(todo):.2f}s/specimen, {workers} workers)")
    print(f"[{view}] gamut metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"))
    parser.add_argument(
        "--metadata-dir",
        default=str(Path(__file__).resolve().parents[1] / "runs" / "a7_gamut_metadata"),
        help="parent folder for per-view gamut metadata CSVs (saved as <metadata-dir>/<view>.csv)",
    )
    parser.add_argument("--views", nargs="+", default=list(VIEWS), choices=VIEWS)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N specimens per view (smoke test)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with open(args.config) as f:
        paths_cfg = yaml.safe_load(f)
    dataset_dir = Path(paths_cfg["dataset_dir"])
    dataset_a7_dir = Path(paths_cfg["dataset_a7_dir"])
    source_root = _check_back_leaf_and_get_parent(dataset_dir, "dataset_dir")
    dest_root = _check_back_leaf_and_get_parent(dataset_a7_dir, "dataset_a7_dir")
    metadata_root = Path(args.metadata_dir)

    for view in args.views:
        source_dir = source_root / VIEW_TEMPLATE.format(view=view)
        if not source_dir.is_dir():
            raise FileNotFoundError(f"missing source view folder: {source_dir}")

    for view in args.views:
        build_view(view, source_root, dest_root, metadata_root, args.workers, args.limit)


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
