"""The evaluation Dataset (protocole §7): one item = one (image, condition)
pair, geometry (flip + crop positions) seeded from the image alone so every
condition sees the exact same crops of the exact same image - that's what
makes the comparison paired rather than between independent samples.

df covers all three views (Back/Belly/Side, see src/datasets/folds.py) -
each stem only exists inside its own view's NPY folder, so __getitem__
looks up which view a given stem belongs to before picking that view's
directory set (src/view_paths.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .. import chain
from ..ablations.common import Condition
from ..rng import derive_rng
from ..evaluation.conditions import EvalCondition, fixed_conditions
from ..view_paths import resolve_view_dirs
from .bag_dataset import IMAGENET_MEAN, IMAGENET_STD


class EvalDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: dict,
        a6_manifest: Optional[pd.DataFrame] = None,
    ):
        self.stems = df["stem"].to_numpy()
        self.habitats = np.stack(df["habitat"].to_numpy())
        self.stem_to_row = {s: i for i, s in enumerate(self.stems)}
        self.stem_to_view = dict(zip(df["stem"], df["view"]))
        # one directory set per view (Back/Belly/Side) - each item picks the
        # one matching its own stem's view, since a stem only exists inside
        # its own view's NPY folder.
        self.view_dirs = resolve_view_dirs(cfg)

        self._fixed = fixed_conditions()
        self._items = self._build_index(a6_manifest)

    def _build_index(self, a6_manifest: Optional[pd.DataFrame]):
        items = []
        for stem in self.stems:
            for ec in self._fixed:
                items.append((stem, ec, None))
        if a6_manifest is not None:
            from ..ablations.common import Condition as _C
            from .. import color as _color

            manifest_by_stem = a6_manifest.set_index("stem")
            a6_condition = _C(active=frozenset({"A6"}))
            for stem in self.stems:
                if stem not in manifest_by_stem.index:
                    continue
                row = manifest_by_stem.loc[stem]
                for category in _color.DELHEY_CHROMATIC_CATEGORIES:
                    if bool(row.get(category, False)):
                        ec = EvalCondition(f"A6_{category}", a6_condition)
                        items.append((stem, ec, category))
        return items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int):
        stem, ec, a6_category = self._items[index]
        dirs = self.view_dirs[self.stem_to_view[stem]]
        geometry_rng = derive_rng("eval-geometry", stem)
        condition_rng = derive_rng("eval-condition", stem, ec.name)

        crops, meta = chain.build_bag(
            dirs["dataset_dir"],
            dirs["dataset_a1_dir"],
            dirs["dataset_a2_dir"],
            dirs["dataset_a7_dir"],
            stem,
            ec.condition,
            condition_rng,
            fixed_a3_side=ec.a3_side,
            fixed_a4_params=ec.a4_params,
            fixed_a6_category=a6_category,
            positions_rng=geometry_rng,
            dataset_a6_class_dir=dirs["dataset_a6_class_dir"],
        )

        crops = crops.astype(np.float32) / 255.0
        crops = (crops - IMAGENET_MEAN) / IMAGENET_STD
        crops = torch.from_numpy(crops).permute(0, 3, 1, 2).contiguous()

        row = self.stem_to_row[stem]
        habitat = torch.from_numpy(self.habitats[row].astype(np.float32))

        return {
            "crops": crops,
            "habitat": habitat,
            "stem": stem,
            "condition_name": ec.name,
        }
