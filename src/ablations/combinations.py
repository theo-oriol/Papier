"""The 36-condition tirage (protocole §4).

Two draws per sac, every epoch: how many ablations (0-3, with probabilities
0.20/0.40/0.30/0.10), then which ones - uniformly among the combinations of
that size that don't contain more than one of the four CIELAB-exclusive
ablations (A1/A2/A6/A7).

The counts below (1, 7, 15, 13 combinations for K=0..3, 36 total) are a
direct consequence of that exclusion rule, not something chosen separately -
see the module-level checks in tests/ or just call `admissible_by_size()`
and count them.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from .common import ALL_CODES, Condition, is_admissible

K_PROBABILITIES = {0: 0.20, 1: 0.40, 2: 0.30, 3: 0.10}


def admissible_by_size() -> Dict[int, List[frozenset]]:
    """
    Retourne la liste de pool d'ablations compatible par nombre d'ablation (0,1,2,3)
    """
    by_size: Dict[int, List[frozenset]] = {k: [] for k in K_PROBABILITIES}
    for k in K_PROBABILITIES:
        for combo in combinations(ALL_CODES, k):
            # c'est un set imuable 
            codes = frozenset(combo)
            if is_admissible(codes):
                by_size[k].append(codes)
    return by_size


_ADMISSIBLE_BY_SIZE = admissible_by_size()


def draw_condition(rng: np.random.Generator, enabled_codes: Optional[FrozenSet[str]] = None) -> Condition:
    """`enabled_codes`, if given, restricts the draw to admissible
    conditions using only codes from that set (rejection sampling on the
    same rng - still deterministic for a given seed, just consumes a few
    more draws whenever a disallowed code comes up). None (the default)
    draws from the full protocol pool exactly as protocole §4 specifies -
    real training (Trainer/BagDataset without an explicit enabled_codes)
    always uses that default. This parameter exists for demos/notebooks
    that want to avoid ablations whose offline cache isn't built yet (e.g.
    A7 - see notebooks/02_training_dataloader_demo.ipynb), not for
    production runs: excluding a code changes the effective marginal
    probabilities away from the protocol's own table.
    """
    sizes = list(K_PROBABILITIES.keys())
    probs = list(K_PROBABILITIES.values())
    # Choisi la pool d'ablation aléatoirement entre 0 et 3 (en utilisant K_PROBABILITIES)
    k = int(rng.choice(sizes, p=probs))
    pool = _ADMISSIBLE_BY_SIZE[k]
    # Choisi la combinaison d'ablation aléatoirement dans le pool choisi précedemment 
    codes = pool[int(rng.integers(len(pool)))]
    # Condition est un objet qui stock les conditions selectionné
    condition = Condition(active=codes)

    if enabled_codes is None or condition.active <= enabled_codes:
        return condition

    for _ in range(1000):
        condition = draw_condition(rng)
        if condition.active <= enabled_codes:
            return condition
    raise RuntimeError(
        f"couldn't draw an admissible condition using only {sorted(enabled_codes)} in 1000 tries"
    )


def all_admissible_conditions() -> List[Condition]:
    """All 36 conditions, flattened - used by the notebook and by tests that
    check the marginals table in protocole §4."""
    conditions = []
    for k in sorted(_ADMISSIBLE_BY_SIZE):
        for codes in _ADMISSIBLE_BY_SIZE[k]:
            conditions.append(Condition(active=codes))
    return conditions


def marginal_inclusion_probabilities() -> Dict[str, float]:
    """P(ablation active in a random sac) per code - should reproduce the
    table in protocole §4 (0.140 for the CIELAB four, 0.246 for A3/A4/A5)."""
    totals = {code: 0.0 for code in ALL_CODES}
    for k, pool in _ADMISSIBLE_BY_SIZE.items():
        if not pool:
            continue
        p_each_combo = K_PROBABILITIES[k] / len(pool)
        for codes in pool:
            for code in codes:
                totals[code] += p_each_combo
    return totals
