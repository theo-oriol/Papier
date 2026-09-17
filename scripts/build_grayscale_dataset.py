#!/usr/bin/env python3
"""Build a grayscale copy of the 3-view dataset (Back/Belly/Side), mirroring
the source's own layout - one output folder of its own, containing the same
three NEW_Segmented-Aves-{view}-NPY subfolders, one file per specimen with
the same three extensions (.rgb.zst, .cand_128.zst, .cand_224.zst).

"Grayscale" here is the project's own definition, not a generic one: A1's
"gris, L* conserve" (protocole §3) - CIELAB a* = b* = 0, L* untouched -
reused directly from src/ablations/a1_gray_lstar.py so this dataset is
built with the exact same transform already used (and verified) elsewhere
in this project, rather than a second, subtly different grayscale.

Only the .rgb.zst pixels change. The .cand_*.zst candidate-position tables
depend on the mask (which pixels are foreground), not on colour - removing
chroma doesn't move the mask boundary - so they're copied byte-for-byte
instead of recomputed. That's most of the point of doing this offline: the
expensive part (walking the mask at 1px pitch to build those tables) has
already been paid for once, by whoever built the source dataset.

    python scripts/build_grayscale_dataset.py \\
        --source /media/oriol@newcefe.newage.fr/LaCie/Datasets/Familly_split_no_ablation \\
        --dest   /media/oriol@newcefe.newage.fr/LaCie/Datasets/Familly_split_grayscale \\
        --workers 8

    python scripts/build_grayscale_dataset.py --limit 20   # smoke test, defaults below
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import npy_io
from src.ablations import a1_gray_lstar

VIEWS = ("Back", "Belly", "Side")
VIEW_DIR_TEMPLATE = "NEW_Segmented-Aves-{view}-NPY"

_SOURCE_DIR = None
_DEST_DIR = None


def _init_worker(source_dir: str, dest_dir: str) -> None:
    global _SOURCE_DIR, _DEST_DIR
    _SOURCE_DIR = Path(source_dir)
    _DEST_DIR = Path(dest_dir)


def _process_one(stem: str) -> None:
    canvas = npy_io.read_canvas(_SOURCE_DIR, stem)
    mask = npy_io.foreground_mask(canvas)
    gray = a1_gray_lstar.build(canvas, mask)
    a1_gray_lstar.write_cache_entry(_DEST_DIR, stem, gray)

    for suffix in (".cand_128.zst", ".cand_224.zst"):
        shutil.copy2(_SOURCE_DIR / f"{stem}{suffix}", _DEST_DIR / f"{stem}{suffix}")


def build_view(view: str, source_root: Path, dest_root: Path, workers: int, limit: int | None) -> None:
    source_dir = source_root / VIEW_DIR_TEMPLATE.format(view=view)
    dest_dir = dest_root / VIEW_DIR_TEMPLATE.format(view=view)
    dest_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.name[: -len(".rgb.zst")] for p in source_dir.glob("*.rgb.zst"))
    if limit:
        stems = stems[:limit]

    already_done = {p.name[: -len(".rgb.zst")] for p in dest_dir.glob("*.rgb.zst")}
    todo = [s for s in stems if s not in already_done]
    print(f"[{view}] {len(stems)} specimens total, {len(already_done)} already done, {len(todo)} to do")
    if not todo:
        return

    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(str(source_dir), str(dest_dir))
    ) as pool:
        list(tqdm(pool.map(_process_one, todo, chunksize=8), total=len(todo), desc=view))

    elapsed = time.time() - t0
    print(f"[{view}] done: {len(todo)} specimens in {elapsed:.0f}s ({elapsed / len(todo):.2f}s/specimen, {workers} workers)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="/media/oriol@newcefe.newage.fr/LaCie/Datasets/Familly_split_no_ablation",
        help="parent folder containing NEW_Segmented-Aves-{Back,Belly,Side}-NPY",
    )
    parser.add_argument(
        "--dest",
        default="/media/oriol@newcefe.newage.fr/LaCie/Datasets/Familly_split_grayscale",
        help="parent folder to create/fill, same layout as --source",
    )
    parser.add_argument("--views", nargs="+", default=list(VIEWS), choices=VIEWS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N specimens per view (smoke test)")
    args = parser.parse_args()

    source_root = Path(args.source)
    dest_root = Path(args.dest)
    for view in args.views:
        if not (source_root / VIEW_DIR_TEMPLATE.format(view=view)).is_dir():
            raise FileNotFoundError(f"missing source view folder: {source_root / VIEW_DIR_TEMPLATE.format(view=view)}")

    for view in args.views:
        build_view(view, source_root, dest_root, args.workers, args.limit)


if __name__ == "__main__":
    main()
