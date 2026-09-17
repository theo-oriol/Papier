"""Shared constants and the small pieces every ablation file needs.

Not an ablation itself — just what A1..A7 have in common, kept here so each
ablation file only contains what's specific to that ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional

# The four ablations that live in the same CIELAB slot and are mutually
# exclusive (protocole §3): at most one of them is active in a given sac,
# and when one of them fires it replaces the "fonction de préparation" step
# entirely (chain.py step 1).
CIELAB_EXCLUSIVE_GROUP: FrozenSet[str] = frozenset({"A1", "A2", "A6", "A7"})

ALL_CODES = ("A1", "A2", "A3", "A4", "A5", "A6", "A7")

# Where in the canonical chain (protocole §2) each ablation is inserted.
# "image" = whole 2024x2024 canvas, before cropping. "crop" = the 224px crop,
# after it has been extracted (and, for the 128px crops, resized).
INSERTION_LEVEL = {
    "A1": "image",  # step 0 — deterministic given the specimen, cached, replaces the source read
    "A2": "image",  # step 0 — same reasoning, cached from the start
    "A3": "crop",   # step 7, last spatial operation
    "A4": "image",  # step 4, before cropping (needs real neighbourhood)
    "A5": "crop",   # step 8, last intensity operation
    "A6": "image",  # step 1 — the only CIELAB-exclusive ablation still computed online (category drawn per sac)
    "A7": "image",  # step 0 — deterministic given the specimen, cached, replaces the source read
}

# Whether the ablation's parameters are drawn once per sac (shared by the 16
# crops) or once per crop (protocole "Ce qui est tiré par sac, ce qui est
# tiré par crop"). A5's target slope is jittered (+/- a5_spectral_slope.
# ALPHA_JITTER) around the crop size's reference value, drawn fresh per
# crop rather than fixed - see that module's docstring for why.
DRAWN_PER = {
    "A1": None,
    "A2": None,
    "A3": "crop",
    "A4": "sac",
    "A5": "crop",
    "A6": "sac",
    "A7": None,
}


@dataclass(frozen=True)
class Condition:
    """Which ablations are active for one sac. An empty set is the
    reference condition. Order never matters — it's a set, not a list — the
    canonical chain in chain.py always applies things in the same fixed
    pipeline order regardless of draw order."""

    active: FrozenSet[str] = field(default_factory=frozenset)

    def __contains__(self, code: str) -> bool:
        return code in self.active

    def __len__(self) -> int:
        return len(self.active)

    def __str__(self) -> str:
        return "reference" if not self.active else "+".join(sorted(self.active))

    @property
    def cielab_member(self) -> Optional[str]:
        """The single active CIELAB-exclusive ablation, if any."""
        """@Property permet de faire c.cielab_member au lieu de c.cielab_member()"""
        hit = self.active & CIELAB_EXCLUSIVE_GROUP
        return next(iter(hit)) if hit else None


def is_admissible(codes: FrozenSet[str]) -> bool:
    """protocole §4: a combination is admissible iff it contains at most one
    of the four mutually-exclusive CIELAB ablations."""
    return len(codes & CIELAB_EXCLUSIVE_GROUP) <= 1
