"""Which chromatic categories are present on each specimen - needed to build
the A6 evaluation grid (protocole §7: "une par categorie chromatique, en ne
comptant que les images ou la categorie est presente").

This is a *whole-bird* property (computed once on the full canvas, same
CIELAB classification used by A6 itself), independent of fold or crop, so
it's cached once to a CSV and reused everywhere instead of being
recomputed. Building it used to be the single slowest thing in this
project - classify_delhey_rgb() on a 2024x2024 canvas is the same expensive
call that makes A6 the bottleneck in the timing benchmark
(scripts/benchmark_ablations.py) - which is why compute_presence_row() now
prefers reading the same classification straight from the offline cache
(build_classification() / scripts/build_a6_classification_cache.py) over
reclassifying, falling back to the online classifier per-specimen only
where that cache is still missing an entry (same fallback chain.py's A6
fast path uses, so a partially-built cache never breaks this either).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from .. import npy_io
from ..ablations import a6_colour_removal
from .. import color

MANIFEST_COLUMNS = ["stem"] + list(color.DELHEY_CHROMATIC_CATEGORIES)


def compute_presence_row(dataset_dir: Path, stem: str, dataset_a6_class_dir: Optional[Path] = None) -> dict:
    present = None
    if dataset_a6_class_dir is not None:
        try:
            labels = a6_colour_removal.read_classification_cache_entry(dataset_a6_class_dir, stem)
            present = set(a6_colour_removal.categories_present_from_labels(labels))
        except FileNotFoundError:
            present = None

    if present is None:
        canvas = npy_io.read_canvas(dataset_dir, stem)
        mask = npy_io.foreground_mask(canvas)
        present = set(a6_colour_removal.categories_present(canvas, mask))

    row = {"stem": stem}
    row.update({cat: (cat in present) for cat in color.DELHEY_CHROMATIC_CATEGORIES})
    return row


def load_manifest(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"A6 presence manifest missing: {path}\n"
            "Run scripts/build_a6_presence_manifest.py first."
        )
    return pd.read_csv(path)


def categories_present_for(manifest: pd.DataFrame, stem: str) -> List[str]:
    row = manifest.loc[manifest["stem"] == stem]
    if row.empty:
        return []
    row = row.iloc[0]
    return [cat for cat in color.DELHEY_CHROMATIC_CATEGORIES if bool(row[cat])]
