"""Backbone + MIL pooling + two heads, wired together (protocole §1).

Input is a batch of sacs, each a stack of 16 crops. The backbone runs on
all (B*16) crops flattened into one batch — it doesn't need to know about
the bag structure, only the attention pooling does.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import build_backbone
from .heads import Heads
from .mil import GatedAttentionPool


class BagModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.backbone = build_backbone(cfg)
        feat_dim = cfg["feat_dim"]
        self.pool = GatedAttentionPool(feat_dim)
        self.heads = Heads(feat_dim, cfg["num_habitats"], dropout=cfg["head_dropout"])

    def forward(self, crops: torch.Tensor):
        """crops: (B, N, 3, 224, 224) -> cls_logits, reg_logits, attention."""
        b, n = crops.shape[:2]
        flat = crops.reshape(b * n, *crops.shape[2:])
        features = self.backbone(flat).reshape(b, n, -1)
        pooled, attention = self.pool(features)
        cls_logits, reg_logits = self.heads(pooled)
        return cls_logits, reg_logits, attention

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
