"""What gets written to a run directory so the run is traceable later:
exactly which config produced it, which code, on what data, when. Nothing
here is used by training itself — it's a record, not a control path.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import torch


def _git_commit(repo_dir: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except FileNotFoundError:
        return None


def build_manifest(cfg: Dict[str, Any], config_path: str, project_root: Path) -> Dict[str, Any]:
    return {
        "run_name": cfg["run_name"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "seed": cfg["seed"],
        "fold": cfg["fold"],
        "sham": cfg["sham"],
        "git_commit": _git_commit(project_root),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "hostname": platform.node(),
        "dataset_dir": cfg["dataset_dir"],
        "fold_dir": cfg["fold_dir"],
    }


def write_manifest(manifest: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
