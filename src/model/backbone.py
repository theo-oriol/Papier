"""DINOv3-L backbone, frozen except LoRA adapters (protocole §1).

Loaded from a local torch.hub checkout, same pattern already used elsewhere
on this machine (bird_project/src/models/backbone.py) - torch.hub.load with
source="local" against a cloned facebookresearch/dinov3 repo, weights from a
local .pth (DINOv3 checkpoints are gated on the HF hub, so this avoids
needing network access / a token at train time).

dinov3's forward() returns the CLS token directly when its `.head` is the
default nn.Identity (which it is here) - that CLS embedding is h_k, the
per-crop feature the MIL attention in model/mil.py pools over.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


def build_backbone(cfg: dict) -> nn.Module:
    backbone_name = cfg["backbone"]["name"]
    weights_path = cfg["dinov3_weights"][backbone_name]

    backbone = torch.hub.load(
        cfg["dinov3_repo"],
        backbone_name,
        source="local",
        weights=weights_path,
    )

    for param in backbone.parameters():
        param.requires_grad = False

    lora_cfg = cfg["backbone"]["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )
    backbone = get_peft_model(backbone, peft_config)
    return backbone
