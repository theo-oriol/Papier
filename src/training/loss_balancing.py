"""Adaptive KL/BCE loss weighting - a deliberate departure from protocole
§1's fixed 0.5/0.5 "pondération KL / BCE": once per epoch, reweight the two
loss terms towards whichever one has stalled, instead of leaving both
fixed for the whole run.

Concretely (EMA-smoothed loss ratio, evaluated once per epoch t):

    r_i(t) = L~_i(t-1) / L~_i(t-1-k)
    w_i(t) = softmax(r_i(t) / T)      over the loss terms i
    w_i(t) clipped into [w_min, w_max]

L~_i is an EMA of term i's raw per-epoch loss (persistence `ema_decay`,
close to 1 = slow to move = "inertia"). r_i close to 1 means that term's
smoothed loss barely moved over the last k epochs ("stalled"); r_i > 1
means it got worse. softmax is monotonic increasing in its input, so no
sign flip is needed: a stalled or worsening term (larger r_i) already gets
the larger weight. Temperature T controls how aggressively weight shifts
towards the stalled term - low T is closer to winner-take-all, high T is
closer to an even split. The band clip (symmetric around 0.5 for this
two-term setup - clipping one term's weight to w automatically puts the
other at 1-w, so the pair always sums to 1) guarantees neither term is
ever fully abandoned.

Before `lag_epochs` epochs of history exist, there is no k-epochs-ago
value to compare against, so update() returns an even split - the same
0.5/0.5 protocole §1 prescribes, used here only as a bootstrap default
rather than a fixed choice for the whole run.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, Optional, Tuple


class AdaptiveLossBalancer:
    def __init__(
        self,
        names: Tuple[str, str] = ("kl", "bce"),
        ema_decay: float = 0.9,
        lag_epochs: int = 5,
        temperature: float = 0.1,
        weight_band: Tuple[float, float] = (0.2, 0.8),
    ):
        if len(names) != 2:
            raise ValueError("AdaptiveLossBalancer only supports exactly two loss terms")
        if not 0.0 < ema_decay < 1.0:
            raise ValueError("ema_decay must lie in (0, 1)")
        if lag_epochs < 1:
            raise ValueError("lag_epochs must be >= 1")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        lo, hi = weight_band
        if not (0.0 <= lo < 0.5 < hi <= 1.0) or abs((lo + hi) - 1.0) > 1e-9:
            raise ValueError("weight_band must be a (low, high) pair symmetric around 0.5, e.g. (0.2, 0.8)")

        self.names = names
        self.ema_decay = ema_decay
        self.lag_epochs = lag_epochs
        self.temperature = temperature
        self.weight_band = weight_band

        self._ema: Dict[str, Optional[float]] = {name: None for name in names}
        # holds the last (lag_epochs + 1) EMA values per term - once full,
        # [0] and [-1] are exactly lag_epochs apart, sliding forward every
        # call as new values push old ones out.
        self._history: Dict[str, Deque[float]] = {name: deque(maxlen=lag_epochs + 1) for name in names}

    def update(self, raw_losses: Dict[str, float]) -> Dict[str, float]:
        """Call once per epoch with that epoch's mean raw loss per term.
        Returns the weights to use for the *next* epoch."""
        for name in self.names:
            raw = raw_losses[name]
            prev = self._ema[name]
            ema = raw if prev is None else self.ema_decay * prev + (1.0 - self.ema_decay) * raw
            self._ema[name] = ema
            self._history[name].append(ema)

        a, b = self.names
        if len(self._history[a]) <= self.lag_epochs:
            return {a: 0.5, b: 0.5}

        r_a = self._history[a][-1] / self._history[a][0]
        r_b = self._history[b][-1] / self._history[b][0]

        # two-term softmax(r / T), computed as a sigmoid of the difference
        # for numerical stability (equivalent, avoids exponentiating large
        # r/T directly).
        w_a = 1.0 / (1.0 + math.exp(-(r_a - r_b) / self.temperature))

        lo, hi = self.weight_band
        w_a = min(max(w_a, lo), hi)
        return {a: w_a, b: 1.0 - w_a}
