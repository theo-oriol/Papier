"""Plain YAML configs, no framework. A run config lists other YAML files to
merge in under `includes` (paths.yaml, model.yaml, optim.yaml) plus its own
run-specific keys (fold, sham, seed...). load_config() flattens all of that
into one dict and nothing more — dot-access, defaults, config composition
beyond a flat merge are deliberately not features here, so a reader can see
the whole resolved config by looking at one dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    merged: Dict[str, Any] = {}
    for included in cfg.pop("includes", []):
        included_path = config_path.parent / included
        with open(included_path) as f:
            merged.update(yaml.safe_load(f))
    merged.update(cfg)
    return merged


def save_resolved_config(cfg: Dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
