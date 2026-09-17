#!/usr/bin/env python3
"""Build a "grayscale, luminance-normalized" copy of the 3-view dataset -
same idea as scripts/build_grayscale_dataset.py (own output folder, same
NEW_Segmented-Aves-{view}-NPY layout, .cand_*.zst copied unchanged), but
using SHINE's lumMatch + histMatch on L* (Willenbockel, Sadr, Fiset, Horne,
Gosselin & Tanaka 2010) instead of A1's per-image achromatic_raw: every
specimen's L* distribution is remapped to match one common target
distribution for the whole population, not just recentred on its own mean.

The core maths (CIELAB conversion, lumMatch, the exact-histogram-match
algorithm) is the fixed version from bird_project/scripts/
generate_meta_data_luminance.py - copied here rather than imported, since
that script reads PNG+alpha and this one reads the .rgb.zst/mask format,
but the statistics themselves are unchanged. The bug that script's docstring
documents (the previous generator assumed every image had the same pixel
count as the first one, silently reading uninitialised memory for any
image with more pixels than that) is fixed here the same way: the target
is a density, and per-image counts are recomputed from it (largest-
remainder method) for that image's own N.

Unlike A1/A2/A7, this one genuinely needs two passes over the whole
population before a single output image can be written - the target
distribution is computed *from* the dataset, not fixed by the protocol -
so this script always runs, per view:
    1. stats pass    - per-specimen (N, mean, std) -> population M, S
    2. target pass   - per-specimen histogram of its own lum-matched L*,
                        accumulated (image-weighted) into one target density
    3. apply pass     - per-specimen lumMatch + exact histMatch to the
                        target, embarrassingly parallel once M/S/target
                        are fixed
Metadata (M, S, target density, per-image diagnostics) is saved to
runs/grayscale_norm_lum_metadata/<view>/ either way, so passes 1-2 need
only be paid once even across repeated or interrupted --dest builds.

    python scripts/build_grayscale_norm_lum_dataset.py \\
        --source /media/oriol@newcefe.newage.fr/LaCie/Datasets/Familly_split_no_ablation \\
        --dest   /media/oriol@newcefe.newage.fr/LaCie/Datasets/Familly_split_grayscale_norm_lum \\
        --workers 8

    python scripts/build_grayscale_norm_lum_dataset.py --limit 20   # smoke test, defaults below
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import npy_io

VIEWS = ("Back", "Belly", "Side")
VIEW_DIR_TEMPLATE = "NEW_Segmented-Aves-{view}-NPY"
N_BINS_DEFAULT = 256
VMAX = 100.0  # L* range

# ---------------------------------------------------------------------------
# CIELAB conversion + SHINE lumMatch/histMatch - same maths as
# bird_project/scripts/generate_meta_data_luminance.py, verified against it
# on real data (see the smoke test in the commit/session that added this).
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
    xyz = _srgb_to_linear(np.asarray(rgb01, dtype=np.float64)) @ _M_RGB2XYZ.T
    fx, fy, fz = [_f(xyz[..., i] / _WHITE[i]) for i in range(3)]
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    xyz = np.stack([_f_inv(fx) * _WHITE[0], _f_inv(fy) * _WHITE[1], _f_inv(fz) * _WHITE[2]], -1)
    return np.clip(_linear_to_srgb(xyz @ _M_XYZ2RGB.T), 0.0, 1.0)


def lum_match(values: np.ndarray, M: float, S: float, vmax: float, clip: bool = True) -> np.ndarray:
    m, s = values.mean(), values.std()
    if s < 1e-8:
        return np.full_like(values, M)
    out = (values - m) / s * S + M
    return np.clip(out, 0.0, vmax) if clip else out


def quantise(values: np.ndarray, n_bins: int, vmax: float) -> np.ndarray:
    step = vmax / (n_bins - 1)
    return np.clip(np.rint(values / step), 0, n_bins - 1).astype(np.int64)


def counts_from_pdf(pdf: np.ndarray, n: int) -> np.ndarray:
    """Integer counts summing to exactly n, proportional to pdf (largest-
    remainder method) - this is the fix: counts are recomputed for this
    image's own N instead of assuming every image has the same N."""
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


def hist_match_exact(values: np.ndarray, pdf: np.ndarray, vmax: float, rng=None) -> np.ndarray:
    n = values.size
    counts = counts_from_pdf(pdf, n)
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
# I/O for this project's NPY format (in place of the reference script's
# PIL+PNG load_channel/standardise_image)
# ---------------------------------------------------------------------------

def _l_star_of(canvas: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lab = rgb_to_lab(canvas.astype(np.float64)[mask] / 255.0)
    return lab[..., 0]


# ---------------------------------------------------------------------------
# Phase 1: per-specimen (N, mean, std) -> population M, S
# ---------------------------------------------------------------------------

_SOURCE_DIR = None


def _init_stats_worker(source_dir: str) -> None:
    global _SOURCE_DIR
    _SOURCE_DIR = Path(source_dir)


def _stats_one(stem: str) -> dict:
    canvas = npy_io.read_canvas(_SOURCE_DIR, stem)
    mask = npy_io.foreground_mask(canvas)
    values = _l_star_of(canvas, mask)
    return {"stem": stem, "n_pixels": int(values.size), "mean": float(values.mean()), "std": float(values.std())}


# ---------------------------------------------------------------------------
# Phase 2: per-specimen lum-matched histogram, accumulated into one target
# ---------------------------------------------------------------------------

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
    values = _l_star_of(canvas, mask)

    raw = lum_match(values, _M, _S, VMAX, clip=False)
    frac_below_0 = float((raw < 0.0).mean())
    frac_above_max = float((raw > VMAX).mean())
    clipped = np.clip(raw, 0.0, VMAX)

    keep = np.ones(clipped.size, dtype=bool) if _CLIP_TARGET else (raw >= 0.0) & (raw <= VMAX)
    hist = np.zeros(_N_BINS, dtype=np.float64)
    if keep.sum() > 0:
        hist = np.bincount(quantise(clipped[keep], _N_BINS, VMAX), minlength=_N_BINS) / keep.sum()

    return {
        "stem": stem,
        "hist": hist,
        "frac_below_0": frac_below_0,
        "frac_above_max": frac_above_max,
        "mean_after_lum": float(clipped.mean()),
        "std_after_lum": float(clipped.std()),
    }


# ---------------------------------------------------------------------------
# Phase 3: apply lumMatch + exact histMatch, write the standardized specimen
# ---------------------------------------------------------------------------

_DEST_DIR = None
_TARGET_PDF = None
_SEED = None


def _init_apply_worker(source_dir: str, dest_dir: str, m: float, s: float, target_pdf: np.ndarray, seed: int) -> None:
    global _SOURCE_DIR, _DEST_DIR, _M, _S, _TARGET_PDF, _SEED
    _SOURCE_DIR = Path(source_dir)
    _DEST_DIR = Path(dest_dir)
    _M, _S, _TARGET_PDF, _SEED = m, s, target_pdf, seed


def _apply_one(stem: str) -> dict:
    canvas = npy_io.read_canvas(_SOURCE_DIR, stem)
    mask = npy_io.foreground_mask(canvas)
    rng = np.random.default_rng((_SEED, hash(stem) % (2**32)))

    rgb01 = canvas.astype(np.float64) / 255.0
    lab = rgb_to_lab(rgb01)
    l_matched = hist_match_exact(lum_match(lab[..., 0][mask], _M, _S, VMAX), _TARGET_PDF, VMAX, rng)

    out_lab = lab.copy()
    out_lab[..., 0][mask] = l_matched
    rgb_out = lab_to_rgb(out_lab)

    out_uint8 = canvas.copy()
    out_uint8[mask] = np.rint(rgb_out[mask] * 255.0).astype(np.uint8)

    dE = np.linalg.norm(rgb_to_lab(out_uint8.astype(np.float64) / 255.0)[mask] - out_lab[mask], axis=-1)

    npy_io.write_array_zst(_DEST_DIR / f"{stem}.rgb.zst", out_uint8)
    for suffix in (".cand_128.zst", ".cand_224.zst"):
        shutil.copy2(_SOURCE_DIR / f"{stem}{suffix}", _DEST_DIR / f"{stem}{suffix}")

    return {"stem": stem, "dE_mean": float(dE.mean()), "dE_max": float(dE.max())}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

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

    levels = np.linspace(0.0, VMAX, n_bins)
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
        value_min=0.0, value_max=VMAX, n_bins=n_bins,
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
    meta_path = metadata_dir / f"meta_data_luminance_{view}.json"
    target_path = metadata_dir / f"target_pdf_luminance_{view}.npy"
    if meta_path.exists() and target_path.exists() and not args.recompute_metadata:
        print(f"[{view}] reusing existing metadata at {meta_path}")
        return json.loads(meta_path.read_text()), np.load(target_path)

    meta, target_pdf, diagnostics = compute_metadata(
        source_dir, stems, args.workers, args.n_bins, args.weight, not args.no_clip_target
    )
    metadata_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    np.save(target_path, target_pdf)
    with open(metadata_dir / f"diagnostics_{view}.csv", "w", newline="") as f:
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
    source_dir = source_root / VIEW_DIR_TEMPLATE.format(view=view)
    dest_dir = dest_root / VIEW_DIR_TEMPLATE.format(view=view)
    metadata_dir = metadata_root / view
    dest_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.name[: -len(".rgb.zst")] for p in source_dir.glob("*.rgb.zst"))
    if args.limit:
        stems = stems[: args.limit]

    meta, target_pdf = load_or_compute_metadata(view, source_dir, stems, metadata_dir, args)
    if args.metadata_only:
        return

    already_done = {p.name[: -len(".rgb.zst")] for p in dest_dir.glob("*.rgb.zst")}
    todo = [s for s in stems if s not in already_done]
    print(f"[{view}] {len(stems)} specimens total, {len(already_done)} already done, {len(todo)} to do")
    if not todo:
        return

    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_apply_worker,
        initargs=(str(source_dir), str(dest_dir), meta["M"], meta["S"], target_pdf, args.seed),
    ) as pool:
        gamut_rows = list(tqdm(pool.map(_apply_one, todo, chunksize=8), total=len(todo), desc=f"{view} apply"))

    with open(metadata_dir / f"gamut_{view}.csv", "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(gamut_rows[0].keys()))
        if f.tell() == 0:
            writer.writeheader()
        writer.writerows(gamut_rows)

    elapsed = time.time() - t0
    print(f"[{view}] done: {len(todo)} specimens in {elapsed:.0f}s ({elapsed / len(todo):.2f}s/specimen, {args.workers} workers)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="/media/oriol@newcefe.newage.fr/LaCie/Datasets/Familly_split_no_ablation",
        help="parent folder containing NEW_Segmented-Aves-{Back,Belly,Side}-NPY",
    )
    parser.add_argument(
        "--dest",
        default="/media/oriol@newcefe.newage.fr/LaCie/Datasets/Familly_split_grayscale_norm_lum",
        help="parent folder to create/fill, same layout as --source",
    )
    parser.add_argument(
        "--metadata-dir",
        default="/home/oriol@newcefe.newage.fr/Models/paper_code/runs/grayscale_norm_lum_metadata",
        help="where M/S/target_pdf/diagnostics are saved (and reused on a later run)",
    )
    parser.add_argument("--views", nargs="+", default=list(VIEWS), choices=VIEWS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-bins", type=int, default=N_BINS_DEFAULT)
    parser.add_argument("--weight", choices=["image", "pixel"], default="image")
    parser.add_argument(
        "--no-clip-target", action="store_true",
        help="exclude out-of-[0,vmax] pixels from the target accumulation instead of stacking them on the bounds",
    )
    parser.add_argument("--recompute-metadata", action="store_true", help="ignore any saved M/S/target_pdf and redo passes 1-2")
    parser.add_argument("--metadata-only", action="store_true", help="compute and save metadata, skip building the dataset")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N specimens per view (smoke test)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source_root = Path(args.source)
    dest_root = Path(args.dest)
    metadata_root = Path(args.metadata_dir)
    for view in args.views:
        if not (source_root / VIEW_DIR_TEMPLATE.format(view=view)).is_dir():
            raise FileNotFoundError(f"missing source view folder: {source_root / VIEW_DIR_TEMPLATE.format(view=view)}")

    for view in args.views:
        build_view(view, source_root, dest_root, metadata_root, args)


if __name__ == "__main__":
    main()
