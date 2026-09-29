#!/usr/bin/env python3
"""Extract frozen DINOv3-S (ViT-S/16) feature vectors for every specimen's
16 crops, in all three views (Back/Belly/Side), under the "reference"
(no-ablation) condition - no LoRA, no MIL attention pooling, no
classification/regression heads. Just the raw, pretrained backbone's
per-crop CLS embedding, saved to disk.

Crops are built by the exact same code path everything else in this project
uses (src/chain.py's build_bag()), with an empty Condition(). By default
(no --run-id) geometry is seeded the same way run_eval.py's "reference"
condition seeds it (derive_rng("eval-geometry", stem) / derive_rng(
"eval-condition", stem, "reference")) - the identical 16 crops (8x128
resized to 224, 8x224 native) the §7 evaluation grid's own reference
condition sees, still passed through the pipeline's mandatory base
preprocessing (median blur) but with none of A1-A7 applied.

--run-id N folds N into that same derivation (derive_rng("eval-geometry",
"run", N, stem), etc.), producing a DIFFERENT but still reproducible flip +
crop-position draw per run - for repeating the whole extraction several
times with genuinely different crops of the same images, e.g. to test how
sensitive downstream results are to which particular crops got drawn.
Omitting --run-id reproduces the original (pre-multi-run) crops exactly,
for anyone who already has that first extraction and doesn't want it to
change. Each --run-id gets its own output subfolder (run{N}/) so repeated
runs accumulate side by side instead of overwriting each other.

configs/paths.yaml's dataset_dir/dataset_a1_dir/etc only name the Back view
(protocole §2, "vue Back uniquement" - the one the training/eval pipeline
reads), but the source dataset and every ablation cache actually hold all
three views as sibling folders under their own parent
(NEW_Segmented-Aves-{view}-NPY) - same derivation build_a1_cache.py /
build_a7_cache.py use. This script builds all three by default.

One output file per specimen per view: {stem}.dinov3s_features.zst, a
zstd-compressed (16, 384) float32 array (16 crops x DINOv3-S's embed_dim),
same container format as the rest of the project's caches (src/npy_io.py).

Resumable: specimens whose output file already exists are skipped, so a
killed run just needs to be re-launched, not restarted from zero.

    python scripts/extract_dinov3s_features.py
    python scripts/extract_dinov3s_features.py --run-id 1
    python scripts/extract_dinov3s_features.py --run-id 2 --batch-size 16 --workers 8
    python scripts/extract_dinov3s_features.py --run-id 3 --views Belly Side   # skip an already-done Back
    python scripts/extract_dinov3s_features.py --run-id 4 --limit 200   # smoke test, all views
    python scripts/extract_dinov3s_features.py --run-id 1 --out-dir /some/other/place
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
VIEWS = ("Back", "Belly", "Side")
VIEW_TEMPLATE = "NEW_Segmented-Aves-{view}-NPY"


def _check_back_leaf_and_get_parent(path: Path, config_key: str) -> Path:
    """`path` must be a 'NEW_Segmented-Aves-Back-NPY' folder directly under
    its parent - that parent is where the Belly/Side siblings are derived
    from. Same check build_a1_cache.py/build_a7_cache.py use; raises loudly
    instead of silently building in the wrong place."""
    expected = path.parent / VIEW_TEMPLATE.format(view="Back")
    if path != expected:
        raise ValueError(
            f"configs/paths.yaml {config_key} ({path}) isn't a "
            f"'{VIEW_TEMPLATE.format(view='Back')}' folder directly under its parent - "
            f"can't derive the Belly/Side sibling folders from it."
        )
    return path.parent


class ReferenceCropsDataset(Dataset):
    """One item = one specimen's 16 reference-condition crops (one view),
    normalized and ready for the backbone - everything except the forward
    pass itself, so that work can run in parallel DataLoader workers while
    the GPU stays busy on the previous batch."""

    def __init__(self, stems: List[str], view_dirs: Dict[str, Optional[Path]], run_id: Optional[int] = None):
        self.stems = stems
        self.dataset_dir = view_dirs["dataset_dir"]
        self.dataset_a1_dir = view_dirs["dataset_a1_dir"]
        self.dataset_a2_dir = view_dirs["dataset_a2_dir"]
        self.dataset_a7_dir = view_dirs["dataset_a7_dir"]
        self.dataset_a6_class_dir = view_dirs["dataset_a6_class_dir"]
        self.reference_condition = Condition()  # empty set = no ablation active
        self.run_id = run_id

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int):
        stem = self.stems[index]
        # same derivation run_eval.py's EvalDataset uses for its "reference"
        # EvalCondition - identical crops to the §7 grid's own baseline -
        # UNLESS run_id is set, in which case it's folded into the hash so a
        # different (but still reproducible) flip + crop draw comes out, for
        # repeated multi-run statistical testing (see module docstring).
        if self.run_id is None:
            geometry_rng = derive_rng("eval-geometry", stem)
            condition_rng = derive_rng("eval-condition", stem, "reference")
        else:
            geometry_rng = derive_rng("eval-geometry", "run", self.run_id, stem)
            condition_rng = derive_rng("eval-condition", "run", self.run_id, stem, "reference")

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


def run_view(
    view: str,
    view_dirs: Dict[str, Optional[Path]],
    out_dir: Path,
    backbone: torch.nn.Module,
    device: torch.device,
    feat_dim: int,
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = view_dirs["dataset_dir"]
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"[{view}] missing source view folder: {dataset_dir}")

    stems = sorted(p.name[: -len(".rgb.zst")] for p in dataset_dir.glob("*.rgb.zst"))
    if args.limit:
        stems = stems[: args.limit]
    already_done = {p.name[: -len(".dinov3s_features.zst")] for p in out_dir.glob("*.dinov3s_features.zst")}
    todo = [s for s in stems if s not in already_done]
    print(f"[{view}] {len(stems)} specimens total, {len(already_done)} already extracted, {len(todo)} to do")
    if not todo:
        return

    meta_path = out_dir / "manifest.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps({
            "backbone": BACKBONE_NAME,
            "weights_path": args.paths_cfg["dinov3_weights"][BACKBONE_NAME],
            "feat_dim": feat_dim,
            "n_crops": N_CROPS,
            "view": view,
            "condition": "reference (no ablation, same crops as run_eval.py's baseline)",
            "run_id": args.run_id,
            "source_dataset_dir": str(dataset_dir),
        }, indent=2))

    dataset = ReferenceCropsDataset(todo, view_dirs, run_id=args.run_id)
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
        for batch in tqdm(loader, desc=f"{view} extract", unit="specimen", unit_scale=args.batch_size):
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
    print(f"[{view}] done: {n_done} specimens in {elapsed:.0f}s ({elapsed / max(n_done, 1):.3f}s/specimen)")


LACIE_DATASETS_ROOT = Path("/media/oriol@newcefe.newage.fr/LaCie/Datasets")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"))
    parser.add_argument("--views", nargs="+", default=list(VIEWS), choices=VIEWS)
    parser.add_argument(
        "--run-id", type=int, default=None,
        help="Folds this into the crop-geometry seed so a repeated run draws DIFFERENT "
             "(but still reproducible) crops of the same images, for multi-run statistical "
             "testing - see module docstring. Omit for the original deterministic reference "
             "crops. Each run-id gets its own run{N}/ output subfolder, so repeats don't "
             "overwrite each other or the no-run-id extraction.",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Defaults to LaCie: /media/oriol@newcefe.newage.fr/LaCie/Datasets/"
             "Familly_split_dinov3s_features[/run{N}] - same sibling-folder convention as "
             "the other Familly_split_* caches, just rooted on LaCie instead of the source "
             "dataset's own drive.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Specimens per forward pass (x16 crops each).")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers building crops in parallel.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N specimens per view (smoke test).")
    parser.add_argument("--use-amp", action="store_true", default=True)
    args = parser.parse_args()

    with open(args.config) as f:
        paths_cfg = yaml.safe_load(f)
    args.paths_cfg = paths_cfg  # threaded through to run_view() for the manifest

    source_root = _check_back_leaf_and_get_parent(Path(paths_cfg["dataset_dir"]), "dataset_dir")
    a1_root = _check_back_leaf_and_get_parent(Path(paths_cfg["dataset_a1_dir"]), "dataset_a1_dir")
    a2_root = _check_back_leaf_and_get_parent(Path(paths_cfg["dataset_a2_dir"]), "dataset_a2_dir")
    a7_root = _check_back_leaf_and_get_parent(Path(paths_cfg["dataset_a7_dir"]), "dataset_a7_dir")
    a6_key = paths_cfg.get("dataset_a6_classification_dir")
    a6_root = _check_back_leaf_and_get_parent(Path(a6_key), "dataset_a6_classification_dir") if a6_key else None

    if args.out_dir:
        out_root = Path(args.out_dir)
    else:
        if not LACIE_DATASETS_ROOT.parent.is_dir():
            raise FileNotFoundError(f"LaCie not reachable at {LACIE_DATASETS_ROOT.parent} - is it mounted?")
        # same sibling-folder convention as Familly_split_A1/A2/A6.../A7,
        # just rooted on LaCie rather than next to the source dataset.
        out_root = LACIE_DATASETS_ROOT / "Familly_split_dinov3s_features"
    if args.run_id is not None:
        out_root = out_root / f"run{args.run_id}"
    print(f"output root: {out_root}  (run_id={args.run_id})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    backbone = build_raw_backbone(paths_cfg, device)
    with torch.no_grad():
        probe = torch.zeros(1, 3, 224, 224, device=device)
        feat_dim = backbone(probe).shape[-1]
    print(f"backbone: {BACKBONE_NAME}  feat_dim: {feat_dim}  frozen params: {sum(p.numel() for p in backbone.parameters()):,}")

    for view in args.views:
        view_dirs = {
            "dataset_dir": source_root / VIEW_TEMPLATE.format(view=view),
            "dataset_a1_dir": a1_root / VIEW_TEMPLATE.format(view=view),
            "dataset_a2_dir": a2_root / VIEW_TEMPLATE.format(view=view),
            "dataset_a7_dir": a7_root / VIEW_TEMPLATE.format(view=view),
            "dataset_a6_class_dir": (a6_root / VIEW_TEMPLATE.format(view=view)) if a6_root else None,
        }
        out_dir = out_root / VIEW_TEMPLATE.format(view=view)
        run_view(view, view_dirs, out_dir, backbone, device, feat_dim, args)


if __name__ == "__main__":
    main()
