#!/usr/bin/env python3
"""CLI entry point for src/evaluation/run_eval.py's protocole §7 grid
evaluation - that module only exposed collect_predictions()/summarise() as
library functions, with no runnable script anywhere in the repo yet.

Two passes, same as the module's own docstring: collect (one DINOv3 forward
pass per (image, condition) pair against the fold's held-out validation
split - the slow part) then summarise (macro AUC/AP, mean KL, bootstrap
delta-AUC vs the "reference" condition - fast, recomputable without
rerunning the model). Both land on disk, so re-running summarise() with a
different --n-bootstrap doesn't require the GPU again.

    python scripts/run_eval.py --config configs/train_sham_fold0.yaml --checkpoint runs/sham_fold0/checkpoints/best.pt
    python scripts/run_eval.py --config configs/train_fold0.yaml --checkpoint runs/main_fold0/checkpoints/best.pt
    python scripts/run_eval.py --config configs/train_fold1.yaml --checkpoint runs/main_fold1/checkpoints/last.pt --out-dir /tmp/eval_fold1_last
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configio import load_config
from src.evaluation.run_eval import collect_predictions, summarise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="The fold's own train_*.yaml (has the right fold/sham/paths already).")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint (usually .../checkpoints/best.pt).")
    parser.add_argument("--fold", type=int, default=None, help="Defaults to the fold already set in --config.")
    parser.add_argument("--out-dir", default=None, help="Defaults to a sibling 'eval' folder next to the checkpoint's run directory.")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    fold = args.fold if args.fold is not None else cfg["fold"]
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    # checkpoint_path is .../<run_name>/checkpoints/<name>.pt - eval/ lands
    # next to checkpoints/, inside the same run directory, by default.
    out_dir = Path(args.out_dir) if args.out_dir else checkpoint_path.parents[1] / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] config={args.config}  fold={fold}  checkpoint={checkpoint_path}")
    predictions = collect_predictions(cfg, checkpoint_path, fold)
    predictions_path = out_dir / "predictions.parquet"
    predictions.to_parquet(predictions_path)
    print(f"[eval] {len(predictions)} (image, condition) predictions -> {predictions_path}")

    summary = summarise(predictions, args.n_bootstrap, args.seed)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[eval] summary ({len(summary)} conditions) -> {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
