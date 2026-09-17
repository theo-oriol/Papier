#!/usr/bin/env python3
"""Entry point for one training run.

    python train.py --config configs/train_fold0.yaml
    python train.py --config configs/train_sham_fold1.yaml

Everything about *what* to run lives in the config file, not in flags —
that's the traceability contract: a run is fully described by the one YAML
file passed here, and a copy of it (with includes resolved) is written into
the run's own output directory.
"""

from __future__ import annotations

import argparse

from src.configio import load_config
from src.training.run import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to a configs/train_*.yaml file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    trainer = Trainer(cfg, config_path=args.config)
    trainer.fit()


if __name__ == "__main__":
    main()
