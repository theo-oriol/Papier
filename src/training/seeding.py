"""Global determinism for one training run. Doesn't touch the per-sac
ablation RNGs (those are already deterministic functions of (seed, epoch,
index) — see src/rng.py) — this only seeds the libraries that draw from
global state: torch's own init/dropout streams and the DataLoader workers.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    # each DataLoader worker only ever decodes one crop at a time, so torch's
    # default intra-op thread pool (sized to nproc) just adds contention
    # between workers instead of speeding anything up
    torch.set_num_threads(1)
