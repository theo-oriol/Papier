#!/usr/bin/env python3
"""Evaluate one trained fold against the full §7 condition grid.

    python evaluate.py --config configs/eval.yaml \
        --checkpoint runs/main_fold0/checkpoints/best.pt --fold 0

Writes, under runs/<checkpoint's run name>/eval/:
    predictions.csv   one row per (image, condition): raw probabilities
    summary.csv        one row per condition: macro AUC/AP, mean KL, paired
                        mean absolute error vs reference, bootstrap delta-AUC

Combining the three folds' summaries by inverse variance (protocole §7,
last paragraph) is a separate, tiny step — src/evaluation/combine_folds.py —
since it only runs once all three folds have been evaluated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.configio import load_config
from src.evaluation.run_eval import collect_predictions, summarise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    run_dir = checkpoint_path.parents[1]  # .../<run_name>/checkpoints/best.pt -> .../<run_name>
    out_dir = run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = collect_predictions(cfg, checkpoint_path, args.fold)
    predictions.to_pickle(out_dir / "predictions.pkl")  # habitat/prob columns hold arrays, not CSV-friendly

    summary = summarise(predictions, n_bootstrap=cfg["n_bootstrap"], seed=cfg.get("seed", 0))
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nwrote {out_dir / 'summary.csv'} and {out_dir / 'predictions.pkl'}")


if __name__ == "__main__":
    main()
