"""The evaluation Dataset (protocole §7): one item = one (image, condition)
pair, geometry (flip + crop positions) seeded from the image alone so every
condition sees the exact same crops of the exact same image - that's what
makes the comparison paired rather than between independent samples.
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
        self.dataset_dir = Path(cfg["dataset_dir"])
        self.dataset_a1_dir = Path(cfg["dataset_a1_dir"])
        self.dataset_a2_dir = Path(cfg["dataset_a2_dir"])
        self.dataset_a7_dir = Path(cfg["dataset_a7_dir"])
        a6_class_dir = cfg.get("dataset_a6_classification_dir")
        self.dataset_a6_class_dir = Path(a6_class_dir) if a6_class_dir else None

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
        geometry_rng = derive_rng("eval-geometry", stem)
        condition_rng = derive_rng("eval-condition", stem, ec.name)

        crops, meta = chain.build_bag(
            self.dataset_dir,
            self.dataset_a1_dir,
            self.dataset_a2_dir,
            self.dataset_a7_dir,
            stem,
            ec.condition,
            condition_rng,
            fixed_a3_side=ec.a3_side,
            fixed_a4_params=ec.a4_params,
            fixed_a6_category=a6_category,
            positions_rng=geometry_rng,
            dataset_a6_class_dir=self.dataset_a6_class_dir,
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
