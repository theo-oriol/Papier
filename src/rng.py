"""One way to turn a tuple of "what determines this draw" into a numpy RNG.

Used everywhere a random choice needs to be reproducible: at training time a
sac's condition is redrawn every epoch (protocole §4, "tirées en ligne ... à
chaque passage"), so the parts include the epoch and the sac index. At
evaluation time positions and the flip must be IDENTICAL across all 31
conditions for a given image (protocole §7: "deux conditions voient les
mêmes positions de crops et la même symétrie"), so the parts there are just
the image name — never the condition.
"""

from __future__ import annotations

import hashlib

import numpy as np


def derive_rng(*parts: object) -> np.random.Generator:
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    seed = int.from_bytes(digest, "big")
    return np.random.default_rng(seed)
