#!/usr/bin/env python3
"""Check the fold/crosswalk invariants the protocol asserts (§7): exact
partition of the validation images and families across the three folds, no
train/valid leakage at the image or family level. Run this once after
touching configs/paths.yaml or the fold CSVs, before trusting anything else.

    python scripts/check_dataset_integrity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import folds


def main() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"
    with open(config_path) as f:
        paths_cfg = yaml.safe_load(f)

    report = folds.check_partition(paths_cfg)
    print("all partition/family-leak checks passed\n")
    print(f"valid sizes:    {report['valid_sizes']}  (protocole: [12630, 18200, 10157])")
    print(f"valid families: {report['valid_families']}  (protocole: [83, 77, 80])")
    print(f"train sizes:    {report['train_sizes']}")


if __name__ == "__main__":
    main()
