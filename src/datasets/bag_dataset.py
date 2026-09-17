"""The training (and sham) Dataset: one item = one sac of 16 crops, its
condition redrawn fresh every time it's fetched (protocole §4: "tirees en
ligne, par sac, a chaque passage").

`sham=True` reproduces the sham model of protocole §4: same everything, but
the condition is always the reference (no ablation ever drawn) - the only
thing that distinguishes the sham run from the main run is that one line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .. import chain
from ..ablations.combinations import draw_condition
from ..ablations.common import Condition
from ..rng import derive_rng

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class BagDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: dict,
        run_seed: int,
        sham: bool = False,
    ):
        self.stems = df["stem"].to_numpy()
        self.habitats = np.stack(df["habitat"].to_numpy())
        self.dataset_dir = Path(cfg["dataset_dir"])
        self.dataset_a1_dir = Path(cfg["dataset_a1_dir"])
        self.dataset_a2_dir = Path(cfg["dataset_a2_dir"])
        self.dataset_a7_dir = Path(cfg["dataset_a7_dir"])
        # A6's offline classification cache (scripts/build_a6_classification_cache.py):
        # optional and falls back to the online classifier per-specimen when
        # absent, so this being None or incomplete never breaks a run - see
        # chain.build_bag()'s dataset_a6_class_dir docstring.
        a6_class_dir = cfg.get("dataset_a6_classification_dir")
        self.dataset_a6_class_dir = Path(a6_class_dir) if a6_class_dir else None
        self.run_seed = run_seed
        self.sham = sham
        # bumped by the training loop at the start of every epoch, so the
        # same index draws a different condition each pass over the data
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.stems)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> Dict[str, object]:
        stem = self.stems[index]
        # Crée un random generator
        rng = derive_rng(self.run_seed, self.epoch, index, stem)

        condition = Condition() if self.sham else draw_condition(rng)
        crops, meta = chain.build_bag(
            self.dataset_dir, self.dataset_a1_dir, self.dataset_a2_dir, self.dataset_a7_dir, stem, condition, rng,
            dataset_a6_class_dir=self.dataset_a6_class_dir,
        )

        crops = crops.astype(np.float32) / 255.0
        crops = (crops - IMAGENET_MEAN) / IMAGENET_STD
        crops = torch.from_numpy(crops).permute(0, 3, 1, 2).contiguous()  # (16, 3, 224, 224)

        habitat = torch.from_numpy(self.habitats[index].astype(np.float32))
        support = (habitat > 0).float()

        return {
            "crops": crops,
            "habitat": habitat,
            "support": support,
            "stem": stem,
            "condition": str(condition),
        }
