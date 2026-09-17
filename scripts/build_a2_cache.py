#!/usr/bin/env python3
"""Build the offline A2 cache (gris + niveau/contraste standardises).

A2 no longer follows the protocol PDF's fixed (50, 15) target (see
src/ablations/a2_gray_standardized.py's docstring for why) - the target is
now a population mean/std/histogram computed from the dataset itself
(SHINE's lumMatch + histMatch, Willenbockel et al. 2010). That means this
script always does three passes over the data, not one:

    1. stats pass   - per-specimen (N, mean, std) of L* -> population M, S
    2. target pass  - per-specimen lum-matched histogram, accumulated
                        (image-weighted) into one target density
    3. apply pass    - per-specimen lumMatch + exact histMatch to that
                        target, embarrassingly parallel once M/S/target
                        are fixed - this is the only pass that writes the
                        actual cache entries

Metadata (M, S, target_pdf, per-image diagnostics) is saved under
--metadata-dir and reused on a later run unless --recompute-metadata is
passed - passes 1-2 don't need repeating just because a training run was
interrupted partway through pass 3, or because more specimens were added
to the dataset since.

Runs once per view (Back/Belly/Side), each with its own M/S/target - a
population statistic computed from one view's own specimens, not shared
across views. configs/paths.yaml's dataset_dir/dataset_a2_dir only name the
Back view (the one the protocol's own training/eval pipeline reads -
protocole §2, "vue Back uniquement"), but both the source dataset and the
A2 cache actually hold all three views as sibling folders under their own
parent (NEW_Segmented-Aves-{view}-NPY, see scripts/build_grayscale_dataset.py)
- Familly_split_no_ablation for the source, Familly_split_A2 for the cache
(its own top-level folder, not nested inside Familly_split_no_ablation -
reorganised there on 2026-09-16). This script derives both parents from
dataset_dir/dataset_a2_dir and, by default, builds the A2 cache (and
runs/a2_metadata/<view>/) for all three views - the Back one lands at
exactly the path paths.yaml already names, Belly/Side are siblings of it.

    python scripts/build_a2_cache.py --workers 8
    python scripts/build_a2_cache.py --views Belly Side   # skip the already-built Back
    python scripts/build_a2_cache.py --limit 100   # smoke test, all views
    python scripts/build_a2_cache.py --metadata-only   # just inspect M/S/target_pdf first
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import npy_io
from src.ablations import a2_gray_standardized as a2

VIEWS = ("Back", "Belly", "Side")
VIEW_TEMPLATE = "NEW_Segmented-Aves-{view}-NPY"

_SOURCE_DIR = None


def _init_stats_worker(source_dir: str) -> None:
    global _SOURCE_DIR
    _SOURCE_DIR = Path(source_dir)


def _stats_one(stem: str) -> dict:
    canvas = npy_io.read_canvas(_SOURCE_DIR, stem)
    mask = npy_io.foreground_mask(canvas)
    stats = a2.compute_stats(canvas, mask)
    stats["stem"] = stem
    return stats


_M = None
_S = None
_N_BINS = None
_CLIP_TARGET = None


def _init_target_worker(source_dir: str, m: float, s: float, n_bins: int, clip_target: bool) -> None:
    global _SOURCE_DIR, _M, _S, _N_BINS, _CLIP_TARGET
    _SOURCE_DIR = Path(source_dir)
    _M, _S, _N_BINS, _CLIP_TARGET = m, s, n_bins, clip_target


def _target_contribution_one(stem: str) -> dict:
    canvas = npy_io.read_canvas(_SOURCE_DIR, stem)
    mask = npy_io.foreground_mask(canvas)
    contribution = a2.compute_target_contribution(canvas, mask, _M, _S, _N_BINS, _CLIP_TARGET)
    contribution["stem"] = stem
    return contribution


_DATASET_A2_DIR = None
_TARGET_PDF = None
_SEED = None


def _init_apply_worker(source_dir: str, dataset_a2_dir: str, m: float, s: float, target_pdf: np.ndarray, seed: int) -> None:
    global _SOURCE_DIR, _DATASET_A2_DIR, _M, _S, _TARGET_PDF, _SEED
    _SOURCE_DIR = Path(source_dir)
    _DATASET_A2_DIR = Path(dataset_a2_dir)
    _M, _S, _TARGET_PDF, _SEED = m, s, target_pdf, seed


def _apply_one(stem: str) -> dict:
    canvas = npy_io.read_canvas(_SOURCE_DIR, stem)
    mask = npy_io.foreground_mask(canvas)
    rng = np.random.default_rng((_SEED, hash(stem) % (2**32)))
    out_uint8, meta = a2.build(canvas, mask, _M, _S, _TARGET_PDF, rng)
    a2.write_cache_entry(_DATASET_A2_DIR, stem, out_uint8)
    meta["stem"] = stem
    return meta


def compute_metadata(source_dir: Path, stems: list, workers: int, n_bins: int, weight: str, clip_target: bool):
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_stats_worker, initargs=(str(source_dir),)) as pool:
        stats_rows = list(tqdm(pool.map(_stats_one, stems, chunksize=16), total=len(stems), desc="stats pass"))

    n_px = np.array([r["n_pixels"] for r in stats_rows], dtype=np.int64)
    w = np.ones(len(stats_rows)) if weight == "image" else n_px / n_px.sum()
    w = w / w.sum()
    M = float(np.average([r["mean"] for r in stats_rows], weights=w))
    S = float(np.average([r["std"] for r in stats_rows], weights=w))

    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_target_worker, initargs=(str(source_dir), M, S, n_bins, clip_target)
    ) as pool:
        target_rows = list(tqdm(pool.map(_target_contribution_one, stems, chunksize=16), total=len(stems), desc="target pass"))

    acc = np.zeros(n_bins, dtype=np.float64)
    for row, wi in zip(target_rows, w):
        acc += wi * row["hist"]
    target_pdf = acc / acc.sum()

    levels = np.linspace(0.0, a2.VMAX, n_bins)
    M_target = float((target_pdf * levels).sum())
    S_target = float(np.sqrt((target_pdf * (levels - M_target) ** 2).sum()))

    diagnostics = [
        {
            "stem": s["stem"],
            "n_pixels": s["n_pixels"],
            "mean": s["mean"],
            "std": s["std"],
            "frac_below_0": t["frac_below_0"],
            "frac_above_max": t["frac_above_max"],
            "mean_after_lum": t["mean_after_lum"],
            "std_after_lum": t["std_after_lum"],
        }
        for s, t in zip(stats_rows, target_rows)
    ]

    meta = dict(
        channel="L", colour_space="CIELAB D65",
        value_min=0.0, value_max=a2.VMAX, n_bins=n_bins,
        M=M, S=S, weight=weight, n_images=len(stats_rows), clip_target=clip_target,
        M_target=M_target, S_target=S_target,
        atom_at_0=float(target_pdf[0]), atom_at_max=float(target_pdf[-1]),
        frac_images_clipping=float(np.mean([(r["frac_below_0"] + r["frac_above_max"]) > 0 for r in diagnostics])),
        frac_pixels_clipped_max=float(max(r["frac_below_0"] + r["frac_above_max"] for r in diagnostics)),
        n_pixels_min=int(n_px.min()), n_pixels_max=int(n_px.max()),
        n_pixels_median=float(np.median(n_px)), n_pixels_distinct=int(np.unique(n_px).size),
    )
    return meta, target_pdf, diagnostics


def load_or_compute_metadata(view: str, source_dir: Path, stems: list, metadata_dir: Path, args):
    meta_path = metadata_dir / "meta_data_a2.json"
    target_path = metadata_dir / "target_pdf_a2.npy"
    if meta_path.exists() and target_path.exists() and not args.recompute_metadata:
        print(f"[{view}] reusing existing metadata at {meta_path}")
        return json.loads(meta_path.read_text()), np.load(target_path)

    meta, target_pdf, diagnostics = compute_metadata(
        source_dir, stems, args.workers, args.n_bins, args.weight, not args.no_clip_target
    )
    metadata_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    np.save(target_path, target_pdf)
    with open(metadata_dir / "diagnostics_a2.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(diagnostics[0].keys()))
        writer.writeheader()
        writer.writerows(diagnostics)

    print(f"[{view}] M = {meta['M']:.3f}  S = {meta['S']:.3f}  N de {meta['n_pixels_min']} a {meta['n_pixels_max']}")
    print(
        f"[{view}] cible : M* = {meta['M_target']:.3f}  S* = {meta['S_target']:.3f}  "
        f"atome en 0 = {meta['atom_at_0']:.2%}  images clippees = {meta['frac_images_clipping']:.1%}"
    )
    return meta, target_pdf


def build_view(view: str, source_root: Path, dest_root: Path, metadata_root: Path, args) -> None:
    source_dir = source_root / VIEW_TEMPLATE.format(view=view)
    dataset_a2_dir = dest_root / VIEW_TEMPLATE.format(view=view)
    dataset_a2_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = metadata_root / view

    stems = sorted(p.name[: -len(".rgb.zst")] for p in source_dir.glob("*.rgb.zst"))
    if args.limit:
        stems = stems[: args.limit]
    print(f"[{view}] {len(stems)} specimens total")

    meta, target_pdf = load_or_compute_metadata(view, source_dir, stems, metadata_dir, args)
    if args.metadata_only:
        return

    already_done = {p.name[: -len(".rgb.zst")] for p in dataset_a2_dir.glob("*.rgb.zst")}
    todo = [s for s in stems if s not in already_done]
    print(f"[{view}] {len(already_done)} already cached, {len(todo)} to do")
    if not todo:
        return

    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_apply_worker,
        initargs=(str(source_dir), str(dataset_a2_dir), meta["M"], meta["S"], target_pdf, args.seed),
    ) as pool:
        gamut_rows = list(tqdm(pool.map(_apply_one, todo, chunksize=8), total=len(todo), desc=f"{view} apply"))

    with open(metadata_dir / "gamut_a2.csv", "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(gamut_rows[0].keys()))
        if f.tell() == 0:
            writer.writeheader()
        writer.writerows(gamut_rows)

    elapsed = time.time() - t0
    print(f"[{view}] done: {len(todo)} specimens in {elapsed:.0f}s ({elapsed / len(todo):.2f}s/specimen, {args.workers} workers)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"))
    parser.add_argument(
        "--metadata-dir",
        default=str(Path(__file__).resolve().parents[1] / "runs" / "a2_metadata"),
        help="parent folder for per-view M/S/target_pdf/diagnostics (saved under <metadata-dir>/<view>/, reused on a later run)",
    )
    parser.add_argument("--views", nargs="+", default=list(VIEWS), choices=VIEWS)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N specimens per view (smoke test)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-bins", type=int, default=a2.N_BINS)
    parser.add_argument("--weight", choices=["image", "pixel"], default=a2.WEIGHT)
    parser.add_argument(
        "--no-clip-target", action="store_true",
        help="exclude out-of-[0,vmax] pixels from the target accumulation instead of stacking them on the bounds",
    )
    parser.add_argument("--recompute-metadata", action="store_true", help="ignore any saved M/S/target_pdf and redo passes 1-2")
    parser.add_argument("--metadata-only", action="store_true", help="compute and save metadata, skip building the cache")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        paths_cfg = yaml.safe_load(f)
    dataset_dir = Path(paths_cfg["dataset_dir"])
    dataset_a2_dir = Path(paths_cfg["dataset_a2_dir"])
    source_root = _check_back_leaf_and_get_parent(dataset_dir, "dataset_dir")
    dest_root = _check_back_leaf_and_get_parent(dataset_a2_dir, "dataset_a2_dir")
    metadata_root = Path(args.metadata_dir)

    for view in args.views:
        source_dir = source_root / VIEW_TEMPLATE.format(view=view)
        if not source_dir.is_dir():
            raise FileNotFoundError(f"missing source view folder: {source_dir}")

    for view in args.views:
        build_view(view, source_root, dest_root, metadata_root, args)


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
