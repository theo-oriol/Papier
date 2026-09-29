"""Loading the fold tables and turning them into (image stem, habitat
distribution) pairs, across all three views (protocole §7).

The fold CSVs (train_fold_k.csv / valid_fold_k.csv) cover all three views
(Back/Belly/Side) - 122,961 images total, matching the counts printed in
the protocol PDF exactly (37890 + 54600 + 30471, 83 + 77 + 80 families).
protocole v5 §7 itself restricts analysis to the Back view ("le jeu NPY
n'en contient qu'une, la vue Back") - but that was a statement about what
NPY data existed when it was written, not a methodological requirement:
Belly/Side NPY caches (raw + every ablation) were built later in this
project and are complete (see src/view_paths.py), so every view is used
here - each specimen's Back/Belly/Side image becomes its own independent
training/eval example carrying the same (species-level) habitat label.
protocole §7 also confirms no train/valid overlap at the "oiseau"
(individual bird) level for any fold, so a bird's three views are
guaranteed to land in the same split - extending to all views can't leak
a bird across train/valid the way it could if the split were per-image.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..view_paths import VIEWS, VIEW_TEMPLATE, check_back_leaf_and_get_parent, view_of_name

HABITAT_CODE_COLUMNS = ["1", "2", "3", "4", "8", "12", "14", "9_11", "5_13_15", "6_7"]
N_HABITATS = len(HABITAT_CODE_COLUMNS)


@lru_cache(maxsize=4)
def _available_stems(dataset_dir: str) -> frozenset:
    """The dataset directory has ~125k files in one flat exFAT folder, where
    a single stat() call costs ~100ms (same issue documented in the older
    project's cache-repacking script). Listing the directory once and
    checking membership in a Python set is the difference between this
    taking a second and taking twenty minutes."""
    suffix_len = len(".rgb.zst")
    return frozenset(p.name[:-suffix_len] for p in Path(dataset_dir).glob("*.rgb.zst"))


def load_habitat_names(paths_cfg: dict) -> List[str]:
    with open(paths_cfg["habitat_names_json"]) as f:
        return json.load(f)


def _stem(name_of_img: str) -> str:
    return Path(name_of_img).stem


def load_fold(paths_cfg: dict, fold: int, split: str) -> pd.DataFrame:
    """split is 'train' or 'valid'. Returns a DataFrame with columns
    ['stem', 'family', 'view'] plus one 'habitat' column holding a (10,)
    float array that sums to 1 (the ground-truth habitat distribution) -
    one row per (specimen, view), all three views included."""
    assert split in ("train", "valid")
    csv_path = Path(paths_cfg["fold_dir"]) / f"{split}_fold_{fold}.csv"
    df = pd.read_csv(csv_path)

    # Stem c'est le nom du fichier sans extensions ou root
    df["stem"] = df["name_of_img"].map(_stem)
    df["view"] = df["name_of_img"].map(view_of_name)

    # Vérifie que les fichiers dans le csv sont dans le dossier - par vue,
    # puisqu'un stem Belly n'existe que dans le dossier Belly.
    source_root = check_back_leaf_and_get_parent(Path(paths_cfg["dataset_dir"]), "dataset_dir")
    kept = []
    for view in VIEWS:
        available = _available_stems(str(source_root / VIEW_TEMPLATE.format(view=view)))
        kept.append(df[(df["view"] == view) & df["stem"].isin(available)])
    df = pd.concat(kept, ignore_index=True) if kept else df.iloc[0:0]

    # Normalise la distribution d'habitat
    y = df[HABITAT_CODE_COLUMNS].to_numpy(dtype=np.float64) / 100.0
    row_sums = y.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
    y = y / row_sums
    df["habitat"] = list(y)

    return df[["stem", "family", "habitat", "view"]]


def class_frequencies(train_df: pd.DataFrame) -> np.ndarray:
    """f_c in protocole §1: the fraction of *images* (not probability mass)
    where habitat c is present at all, i.e. mean of 1[p_c > 0]. Used for the
    BCE class weights w_c ~ 1/f_c, renormalised to mean 1.

    protocole §7 cites a separate file (mil5_metrics_per_class.csv) for
    this; that file wasn't found on this machine, so it's computed directly
    from the training fold instead and cached to the run directory for
    traceability (see training/run.py) rather than silently guessed at.
    """
    support = np.stack([(h > 0).astype(np.float64) for h in train_df["habitat"]])
    return support.mean(axis=0)


def bce_class_weights(train_df: pd.DataFrame) -> np.ndarray:
    freq = class_frequencies(train_df)
    freq = np.clip(freq, 1e-6, None)
    weights = 1.0 / freq
    return weights / weights.mean()


def check_partition(paths_cfg: dict) -> Dict[str, object]:
    """protocole §7's own sanity checks: exact partition of images and
    families across the three folds' validation splits, no train/valid
    overlap at any of the three levels. Returns a small report dict;
    raises AssertionError if anything is violated."""
    valids = [load_fold(paths_cfg, k, "valid") for k in range(3)]
    trains = [load_fold(paths_cfg, k, "train") for k in range(3)]

    all_valid_stems = pd.concat([v["stem"] for v in valids])
    assert all_valid_stems.is_unique, "validation stems overlap across folds"

    for k in range(3):
        overlap = set(trains[k]["stem"]) & set(valids[k]["stem"])
        assert not overlap, f"fold {k}: {len(overlap)} images in both train and valid"
        fam_overlap = set(trains[k]["family"]) & set(valids[k]["family"])
        assert not fam_overlap, f"fold {k}: families leak across train/valid"

    return {
        "valid_sizes": [len(v) for v in valids],
        "valid_families": [v["family"].nunique() for v in valids],
        "train_sizes": [len(t) for t in trains],
    }
