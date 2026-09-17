"""Repass one fold's validation set through every condition in the §7 grid,
at frozen weights, and summarise the result per condition.

Two passes: first collect every (image, condition) prediction into one
DataFrame (this is the slow part — one DINOv3-L forward pass per sac), then
compute metrics.py's summaries from that DataFrame. Splitting it this way
means the summary can be recomputed (different n_bootstrap, say) without
re-running the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..datasets import folds
from ..datasets.eval_dataset import EvalDataset
from ..model.bagmodel import BagModel
from . import metrics as metrics_mod
from .a6_presence import load_manifest


def load_checkpoint(model: BagModel, checkpoint_path: Path) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    model.backbone.load_state_dict(state["backbone"])
    model.pool.load_state_dict(state["pool"])
    model.heads.load_state_dict(state["heads"])


def collect_predictions(cfg: Dict[str, Any], checkpoint_path: Path, fold: int) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BagModel(cfg).to(device)
    load_checkpoint(model, checkpoint_path)
    model.eval()

    valid_df = folds.load_fold(cfg, fold, "valid")

    a6_manifest_path = Path(cfg["runs_dir"]) / "a6_presence_manifest.csv"
    a6_manifest = load_manifest(a6_manifest_path) if a6_manifest_path.exists() else None
    if a6_manifest is None:
        print(f"[eval] {a6_manifest_path} not found — skipping the A6 conditions "
              f"(run scripts/build_a6_presence_manifest.py to include them)")

    dataset = EvalDataset(valid_df, cfg, a6_manifest=a6_manifest)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    family_by_stem = dict(zip(valid_df["stem"], valid_df["family"]))

    rows = []
    with torch.no_grad():
        for batch in loader:
            crops = batch["crops"].to(device)
            cls_logits, reg_logits, _attention = model(crops)
            cls_prob = torch.sigmoid(cls_logits).cpu().numpy()
            reg_prob = F.softmax(reg_logits, dim=-1).cpu().numpy()
            habitat = batch["habitat"].numpy()

            for i, stem in enumerate(batch["stem"]):
                rows.append(
                    {
                        "stem": stem,
                        "family": family_by_stem[stem],
                        "condition_name": batch["condition_name"][i],
                        "habitat": habitat[i],
                        "cls_prob": cls_prob[i],
                        "reg_prob": reg_prob[i],
                    }
                )
    return pd.DataFrame(rows)


def summarise(predictions: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    reference = predictions[predictions["condition_name"] == "reference"].set_index("stem")

    summaries = []
    for condition_name, group in predictions.groupby("condition_name"):
        group = group.set_index("stem")
        joined = group.join(reference[["reg_prob"]], rsuffix="_ref")
        support = np.stack([(h > 0).astype(int) for h in group["habitat"]])
        cls_prob = np.stack(group["cls_prob"])
        reg_prob = np.stack(group["reg_prob"])
        habitat = np.stack(group["habitat"])
        families = group["family"].to_numpy()

        row = {
            "condition_name": condition_name,
            "n_images": len(group),
            "macro_auc": metrics_mod.macro_auc(support, cls_prob),
            "macro_ap": metrics_mod.macro_average_precision(support, cls_prob),
            "mean_kl": metrics_mod.mean_kl(habitat, reg_prob),
        }
        if condition_name != "reference":
            ref_reg_prob = np.stack(joined["reg_prob_ref"])
            row["mean_abs_error_vs_reference"] = metrics_mod.mean_absolute_error_to_reference(reg_prob, ref_reg_prob)
            ref_support = np.stack([(h > 0).astype(int) for h in reference.loc[group.index, "habitat"]])
            ref_cls_prob = np.stack(reference.loc[group.index, "cls_prob"])
            boot = metrics_mod.bootstrap_delta_auc(support, cls_prob, ref_cls_prob, families, n_bootstrap, rng)
            row.update(boot)
        summaries.append(row)

    return pd.DataFrame(summaries)
