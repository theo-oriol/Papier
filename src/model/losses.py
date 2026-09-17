"""The combined loss:

    L = w_kl * KL(p_true || softmax(z_reg)) + w_bce * mean_c[ w_c * BCE_c(1[p_true>0], sigmoid(z_cls)) ]

protocole §1 ("pondération KL / BCE") prescribes both terms fixed at
0.5/0.5 for the whole run - LAMBDA_KL/LAMBDA_BCE below are exactly that,
used as CombinedLoss's starting weights. Trainer.fit() (src/training/run.py)
now reweights w_kl/w_bce once per epoch instead of leaving them fixed - see
src/training/loss_balancing.py for why and how; set_weights() is what it
calls to apply each epoch's update. With no extra rescaling either way, the
BCE is *averaged* over the 10 classes (not summed, which would put it at
10x the KL's weight), so the two terms are on comparable scales to begin
with, before any reweighting.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LAMBDA_KL = 0.5
LAMBDA_BCE = 0.5


class CombinedLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, lambda_kl: float = LAMBDA_KL, lambda_bce: float = LAMBDA_BCE):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.lambda_kl = lambda_kl
        self.lambda_bce = lambda_bce

    def set_weights(self, lambda_kl: float, lambda_bce: float) -> None:
        self.lambda_kl = lambda_kl
        self.lambda_bce = lambda_bce

    def forward(self, cls_logits, reg_logits, support, habitat):
        log_pred = F.log_softmax(reg_logits, dim=-1)
        # KL(true || pred): sum_c true_c * (log true_c - log pred_c), true's
        # own entropy term is a constant wrt the model so it's dropped —
        # kept here anyway for a loss value that's directly comparable
        # across conditions/runs.
        safe_true = habitat.clamp_min(1e-12)
        kl = (habitat * (safe_true.log() - log_pred)).sum(dim=-1).mean()

        bce_per_class = F.binary_cross_entropy_with_logits(cls_logits, support, reduction="none")
        bce = (bce_per_class * self.class_weights).mean()

        return self.lambda_kl * kl + self.lambda_bce * bce, {"kl": kl.item(), "bce": bce.item()}
