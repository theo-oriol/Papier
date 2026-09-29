"""Resolving the Back/Belly/Side sibling directories from configs/paths.yaml's
Back-leaf entries - the same derivation scripts/build_a1_cache.py,
build_a7_cache.py etc. already duplicate, factored out once here so
folds.py/bag_dataset.py/eval_dataset.py (which now read all three views
instead of Back only) share one implementation instead of three.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

VIEWS = ("Back", "Belly", "Side")
VIEW_TEMPLATE = "NEW_Segmented-Aves-{view}-NPY"


def view_of_name(name: str) -> str:
    """Which view a filename/stem belongs to - the view is embedded in the
    name itself (e.g. '..._Back_...'), which is also what keeps stems
    globally unique across views without needing any extra disambiguation."""
    for view in VIEWS:
        if f"_{view}_" in name:
            return view
    raise ValueError(f"couldn't find a view (Back/Belly/Side) in {name!r}")


def check_back_leaf_and_get_parent(path: Path, config_key: str) -> Path:
    """`path` must be a 'NEW_Segmented-Aves-Back-NPY' folder directly under
    its parent - that parent is where the Belly/Side siblings are derived
    from. Raises loudly instead of silently reading/writing the wrong place
    if configs/paths.yaml ever points this key somewhere else."""
    expected = path.parent / VIEW_TEMPLATE.format(view="Back")
    if path != expected:
        raise ValueError(
            f"configs/paths.yaml {config_key} ({path}) isn't a "
            f"'{VIEW_TEMPLATE.format(view='Back')}' folder directly under its parent - "
            f"can't derive the Belly/Side sibling folders from it."
        )
    return path.parent


def resolve_view_dirs(paths_cfg: dict) -> Dict[str, Dict[str, Optional[Path]]]:
    """Returns {"Back": {"dataset_dir": ..., "dataset_a1_dir": ..., ...},
    "Belly": {...}, "Side": {...}} - one full set of directories per view,
    derived from paths.yaml's Back-leaf entries the same way the offline
    cache-building scripts already do."""
    source_root = check_back_leaf_and_get_parent(Path(paths_cfg["dataset_dir"]), "dataset_dir")
    a1_root = check_back_leaf_and_get_parent(Path(paths_cfg["dataset_a1_dir"]), "dataset_a1_dir")
    a2_root = check_back_leaf_and_get_parent(Path(paths_cfg["dataset_a2_dir"]), "dataset_a2_dir")
    a7_root = check_back_leaf_and_get_parent(Path(paths_cfg["dataset_a7_dir"]), "dataset_a7_dir")
    a6_key = paths_cfg.get("dataset_a6_classification_dir")
    a6_root = check_back_leaf_and_get_parent(Path(a6_key), "dataset_a6_classification_dir") if a6_key else None

    dirs_by_view = {}
    for view in VIEWS:
        leaf = VIEW_TEMPLATE.format(view=view)
        dirs_by_view[view] = {
            "dataset_dir": source_root / leaf,
            "dataset_a1_dir": a1_root / leaf,
            "dataset_a2_dir": a2_root / leaf,
            "dataset_a7_dir": a7_root / leaf,
            "dataset_a6_class_dir": (a6_root / leaf) if a6_root else None,
        }
    return dirs_by_view
