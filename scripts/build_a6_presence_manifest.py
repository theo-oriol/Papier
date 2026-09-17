#!/usr/bin/env python3
"""Build the A6 presence manifest (protocole §7: which chromatic categories
are present on each specimen, needed to know which A6 evaluation passages
apply to which images).

Classifying one 2024x2024 canvas into Delhey categories takes ~12s on this
machine (see scripts/benchmark_ablations.py's fine-grained breakdown), and
there are ~41,000 specimens - at 8 workers that's still roughly 17 hours
if every specimen needs reclassifying. compute_presence_row() avoids that
entirely for any specimen already in the offline classification cache
(scripts/build_a6_classification_cache.py) - reading its precomputed label
map instead - and only falls back to the online classifier for specimens
that cache doesn't have yet, so running this after (or alongside) that
build script is far cheaper than running it alone. It is resumable
(re-running skips stems already in the output CSV) and parallelised across
processes either way.

    python scripts/build_a6_presence_manifest.py --workers 8
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.a6_presence import MANIFEST_COLUMNS, compute_presence_row

_DATASET_DIR = None
_DATASET_A6_CLASS_DIR = None


def _init_worker(dataset_dir: str, dataset_a6_class_dir: str = None) -> None:
    global _DATASET_DIR, _DATASET_A6_CLASS_DIR
    _DATASET_DIR = Path(dataset_dir)
    _DATASET_A6_CLASS_DIR = Path(dataset_a6_class_dir) if dataset_a6_class_dir else None


def _process_one(stem: str) -> dict:
    return compute_presence_row(_DATASET_DIR, stem, dataset_a6_class_dir=_DATASET_A6_CLASS_DIR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N specimens (smoke test)")
    args = parser.parse_args()

    with open(args.config) as f:
        paths_cfg = yaml.safe_load(f)
    dataset_dir = Path(paths_cfg["dataset_dir"])
    dataset_a6_class_dir = paths_cfg.get("dataset_a6_classification_dir")
    out_path = Path(paths_cfg["runs_dir"]) / "a6_presence_manifest.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.name[: -len(".rgb.zst")] for p in dataset_dir.glob("*.rgb.zst"))
    if args.limit:
        stems = stems[: args.limit]

    already_done = set()
    if out_path.exists():
        with open(out_path) as f:
            already_done = {row["stem"] for row in csv.DictReader(f)}
    todo = [s for s in stems if s not in already_done]
    print(f"{len(stems)} specimens total, {len(already_done)} already done, {len(todo)} to do")
    if dataset_a6_class_dir:
        n_cached = sum(1 for _ in Path(dataset_a6_class_dir).glob("*.delhey_labels.zst"))
        print(f"classification cache: {n_cached} specimens ready ({dataset_a6_class_dir})")

    write_header = not out_path.exists()
    t0 = time.time()
    with open(out_path, "a", newline="") as f, ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(str(dataset_dir), dataset_a6_class_dir)
    ) as pool:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in tqdm(pool.map(_process_one, todo, chunksize=4), total=len(todo)):
            writer.writerow(row)
            f.flush()  # every row committed to disk - a killed run loses nothing

    if todo:
        elapsed = time.time() - t0
        print(f"done: {len(todo)} specimens in {elapsed:.0f}s ({elapsed/len(todo):.2f}s/specimen, {args.workers} workers)")


if __name__ == "__main__":
    main()
