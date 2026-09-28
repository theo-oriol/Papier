#!/usr/bin/env python3
"""Extract frozen DINOv3-S (ViT-S/16) feature vectors for every specimen's
16 crops, under the "reference" (no-ablation) condition - no LoRA, no MIL
attention pooling, no classification/regression heads. Just the raw,
pretrained backbone's per-crop CLS embedding, saved to disk.

Crops are built by the exact same code path everything else in this project
uses (src/chain.py's build_bag()), with an empty Condition() and geometry
seeded the same way run_eval.py's "reference" condition seeds it
(derive_rng("eval-geometry", stem) / derive_rng("eval-condition", stem,
"reference")) - so these are the identical 16 crops (8x128 resized to 224,
8x224 native) the §7 evaluation grid's own reference condition sees, still
passed through the pipeline's mandatory base preprocessing (median blur)
but with none of A1-A7 applied.

One output file per specimen: {stem}.dinov3s_features.zst, a zstd-compressed
(16, 384) float32 array (16 crops x DINOv3-S's embed_dim), same container
format as the rest of the project's caches (src/npy_io.py).

Resumable: specimens whose output file already exists are skipped, so a
killed run just needs to be re-launched, not restarted from zero.

    python scripts/extract_dinov3s_features.py
    python scripts/extract_dinov3s_features.py --batch-size 16 --workers 8
    python scripts/extract_dinov3s_features.py --limit 200   # smoke test
    python scripts/extract_dinov3s_features.py --out-dir /some/other/place
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import chain, npy_io
from src.ablations.common import Condition
from src.datasets.bag_dataset import IMAGENET_MEAN, IMAGENET_STD
from src.rng import derive_rng
from src.training.seeding import worker_init_fn

BACKBONE_NAME = "dinov3_vits16"
N_CROPS = 16  # 8x128 (resized to 224) + 8x224, same order build_bag always returns


class ReferenceCropsDataset(Dataset):
    """One item = one specimen's 16 reference-condition crops, normalized
    and ready for the backbone - everything except the forward pass itself,
    so that work can run in parallel DataLoader workers while the GPU stays
    busy on the previous batch."""

    def __init__(self, stems: List[str], paths_cfg: Dict[str, str]):
        self.stems = stems
        self.dataset_dir = Path(paths_cfg["dataset_dir"])
        self.dataset_a1_dir = Path(paths_cfg["dataset_a1_dir"])
        self.dataset_a2_dir = Path(paths_cfg["dataset_a2_dir"])
        self.dataset_a7_dir = Path(paths_cfg["dataset_a7_dir"])
        a6_dir = paths_cfg.get("dataset_a6_classification_dir")
        self.dataset_a6_class_dir = Path(a6_dir) if a6_dir else None
        self.reference_condition = Condition()  # empty set = no ablation active

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int):
        stem = self.stems[index]
        # same derivation run_eval.py's EvalDataset uses for its "reference"
        # EvalCondition - identical crops to the §7 grid's own baseline.
        geometry_rng = derive_rng("eval-geometry", stem)
        condition_rng = derive_rng("eval-condition", stem, "reference")

        crops, _meta = chain.build_bag(
            self.dataset_dir,
            self.dataset_a1_dir,
            self.dataset_a2_dir,
            self.dataset_a7_dir,
            stem,
            self.reference_condition,
            condition_rng,
            positions_rng=geometry_rng,
            dataset_a6_class_dir=self.dataset_a6_class_dir,
        )

        crops = crops.astype(np.float32) / 255.0
        crops = (crops - IMAGENET_MEAN) / IMAGENET_STD
        crops = torch.from_numpy(crops).permute(0, 3, 1, 2).contiguous()  # (16, 3, 224, 224)
        return {"crops": crops, "stem": stem}


def build_raw_backbone(paths_cfg: Dict[str, str], device: torch.device) -> torch.nn.Module:
    """The plain pretrained DINOv3-S, no LoRA/peft wrapping at all (unlike
    src/model/backbone.py's build_backbone(), which exists for training) -
    every weight frozen, eval mode, nothing but the CLS-token forward pass."""
    weights_path = paths_cfg["dinov3_weights"][BACKBONE_NAME]
    backbone = torch.hub.load(
        paths_cfg["dinov3_repo"],
        BACKBONE_NAME,
        source="local",
        weights=weights_path,
    )
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()
    return backbone.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"))
    parser.add_argument("--out-dir", default=None, help="Defaults to a Familly_split_dinov3s_features sibling folder next to the other caches.")
    parser.add_argument("--batch-size", type=int, default=8, help="Specimens per forward pass (x16 crops each).")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers building crops in parallel.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N specimens (smoke test).")
    parser.add_argument("--use-amp", action="store_true", default=True)
    args = parser.parse_args()

    with open(args.config) as f:
        paths_cfg = yaml.safe_load(f)

    dataset_dir = Path(paths_cfg["dataset_dir"])
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        # .../Datasets/Familly_split_no_ablation/NEW_Segmented-Aves-Back-NPY
        # -> .../Datasets/Familly_split_dinov3s_features/NEW_Segmented-Aves-Back-NPY
        # same sibling-folder convention as Familly_split_A1/A2/A6.../A7.
        datasets_root = dataset_dir.parent.parent
        leaf_name = dataset_dir.name
        out_dir = datasets_root / "Familly_split_dinov3s_features" / leaf_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_dir}")

    stems = sorted(p.name[: -len(".rgb.zst")] for p in dataset_dir.glob("*.rgb.zst"))
    if args.limit:
        stems = stems[: args.limit]
    already_done = {p.name[: -len(".dinov3s_features.zst")] for p in out_dir.glob("*.dinov3s_features.zst")}
    todo = [s for s in stems if s not in already_done]
    print(f"{len(stems)} specimens total, {len(already_done)} already extracted, {len(todo)} to do")
    if not todo:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    backbone = build_raw_backbone(paths_cfg, device)
    with torch.no_grad():
        probe = torch.zeros(1, 3, 224, 224, device=device)
        feat_dim = backbone(probe).shape[-1]
    print(f"backbone: {BACKBONE_NAME}  feat_dim: {feat_dim}  frozen params: {sum(p.numel() for p in backbone.parameters()):,}")

    meta_path = out_dir / "manifest.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps({
            "backbone": BACKBONE_NAME,
            "weights_path": paths_cfg["dinov3_weights"][BACKBONE_NAME],
            "feat_dim": feat_dim,
            "n_crops": N_CROPS,
            "condition": "reference (no ablation, same crops as run_eval.py's baseline)",
            "source_dataset_dir": str(dataset_dir),
        }, indent=2))

    dataset = ReferenceCropsDataset(todo, paths_cfg)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )

    use_amp = args.use_amp and device.type == "cuda"
    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="extract", unit="specimen", unit_scale=args.batch_size):
            crops = batch["crops"].to(device, non_blocking=True)  # (B, 16, 3, 224, 224)
            stems_batch = batch["stem"]
            b, n = crops.shape[:2]
            flat = crops.reshape(b * n, *crops.shape[2:])

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                feats = backbone(flat)  # (B*16, feat_dim)
            feats = feats.reshape(b, n, -1).float().cpu().numpy()

            for i, stem in enumerate(stems_batch):
                npy_io.write_array_zst(out_dir / f"{stem}.dinov3s_features.zst", feats[i])
            n_done += b

    elapsed = time.time() - t0
    print(f"done: {n_done} specimens in {elapsed:.0f}s ({elapsed / max(n_done, 1):.3f}s/specimen)")


if __name__ == "__main__":
    main()
