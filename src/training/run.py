"""The training loop itself (protocole §1 "Optimisation" + §4 sham).

One call to Trainer.fit() is one of the six runs the protocol describes:
three folds x {main, sham}. Everything that makes a run reproducible later
gets written to its run directory up front (resolved config + manifest),
and every epoch appends one row to metrics.csv — that's the whole
traceability story, deliberately nothing fancier than files on disk.
"""

from __future__ import annotations

import csv
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..datasets import folds
from ..datasets.bag_dataset import BagDataset
from ..model.bagmodel import BagModel
from ..model.losses import CombinedLoss
from ..configio import save_resolved_config
from .loss_balancing import AdaptiveLossBalancer
from .manifest import build_manifest, write_manifest
from .seeding import seed_everything, worker_init_fn

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Trainer:
    def __init__(self, cfg: Dict[str, Any], config_path: str):
        # Crée les dossiers du modèle 
        self.cfg = cfg
        self.run_dir = Path(cfg["runs_dir"]) / cfg["run_name"]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)

        seed_everything(cfg["seed"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Copie la config
        save_resolved_config(cfg, self.run_dir / "config.yaml")
        # Crée un manisfest avec des infos config et des méta info sur l'env
        manifest = build_manifest(cfg, config_path, PROJECT_ROOT)

        # Load la liste des fichiers et les labels 
        train_df = folds.load_fold(cfg, cfg["fold"], "train")
        # Calcule la pondération pour la loss en prenant l'inverse de la fréquence d'apparition
        class_weights = torch.tensor(folds.bce_class_weights(train_df), dtype=torch.float32)
        manifest["train_size"] = len(train_df)
        manifest["bce_class_weights"] = class_weights.tolist()
        # Save le manifest
        write_manifest(manifest, self.run_dir / "manifest.json")

        # Crée les crop et applique les ablations. C'est le cerveau du dataloader
        self.dataset = BagDataset(
            train_df,
            cfg=cfg,
            run_seed=cfg["seed"],
            sham=cfg["sham"],
        )
        # the DataLoader hands out micro-batches; the Trainer accumulates
        # gradients over several of them to reach the config's real
        # batch_size without needing that many sacs in GPU memory at once
        # (see configs/optim.yaml's comment on micro_batch_size).
        if cfg["batch_size"] % cfg["micro_batch_size"] != 0:
            raise ValueError("batch_size must be a multiple of micro_batch_size")
        self.accumulation_steps = cfg["batch_size"] // cfg["micro_batch_size"]

        self.loader = DataLoader(
            self.dataset,
            batch_size=cfg["micro_batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            worker_init_fn=worker_init_fn,
            drop_last=True,
            pin_memory=True,
        )

        self.model = BagModel(cfg).to(self.device)
        self.loss_fn = CombinedLoss(class_weights).to(self.device)
        self.loss_balancer = AdaptiveLossBalancer(
            names=("kl", "bce"),
            ema_decay=cfg["loss_balance_ema_decay"],
            lag_epochs=cfg["loss_balance_lag_epochs"],
            temperature=cfg["loss_balance_temperature"],
            weight_band=tuple(cfg["loss_balance_weight_band"]),
        )
        self.optimizer = Adam(
            self.model.trainable_parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg["epochs_max"])
        self.use_amp = cfg["use_amp"] and self.device.type == "cuda"

        # Early-stop bookkeeping lives on self, not as fit()-local variables,
        # so a resumed run can restore it instead of re-deciding "stalled?"
        # from a blank slate (see _load_checkpoint).
        self.start_epoch = 1
        self.recent_losses: deque = deque(maxlen=cfg["early_stop_smoothing"])
        self.best_smoothed = float("inf")
        self.epochs_without_improvement = 0

        self.metrics_path = self.run_dir / "metrics.csv"
        last_ckpt = self.run_dir / "checkpoints" / "last.pt"
        resumed = last_ckpt.exists() and self._load_checkpoint(last_ckpt)
        if not resumed:
            with open(self.metrics_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["epoch", "train_loss", "train_loss_smoothed", "kl", "bce", "w_kl", "w_bce", "lr", "seconds"]
                )

    def _run_epoch(self, epoch: int) -> Dict[str, float]:
        self.dataset.set_epoch(epoch)
        self.model.train()
        total_loss = total_kl = total_bce = 0.0
        n_micro_batches = 0

        self.optimizer.zero_grad(set_to_none=True)
        progress = tqdm(self.loader, desc=f"epoch {epoch}", unit="sac", leave=False)
        for step, batch in enumerate(progress):
            crops = batch["crops"].to(self.device, non_blocking=True)
            habitat = batch["habitat"].to(self.device, non_blocking=True)
            support = batch["support"].to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                cls_logits, reg_logits, _attention = self.model(crops)
                loss, parts = self.loss_fn(cls_logits, reg_logits, support, habitat)

            # scale down so accumulated gradients average to the same thing
            # a single batch_size-sized step would have produced
            (loss / self.accumulation_steps).backward()

            is_last = step == len(self.loader) - 1
            if (step + 1) % self.accumulation_steps == 0 or is_last:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            total_kl += parts["kl"]
            total_bce += parts["bce"]
            n_micro_batches += 1
            progress.set_postfix(loss=total_loss / n_micro_batches, refresh=False)

        return {
            "train_loss": total_loss / n_micro_batches,
            "kl": total_kl / n_micro_batches,
            "bce": total_bce / n_micro_batches,
        }

    def fit(self) -> None:
        patience = self.cfg["early_stop_patience"]

        for epoch in range(self.start_epoch, self.cfg["epochs_max"] + 1):
            t0 = time.time()
            # loss_fn.lambda_kl/lambda_bce are whatever the previous
            # iteration's balancer update left them at (0.5/0.5 to start) -
            # logged below as the weights actually used *this* epoch, then
            # updated afterwards for the next one.
            w_kl, w_bce = self.loss_fn.lambda_kl, self.loss_fn.lambda_bce
            epoch_stats = self._run_epoch(epoch)
            self.scheduler.step()
            elapsed = time.time() - t0

            self.recent_losses.append(epoch_stats["train_loss"])
            smoothed = sum(self.recent_losses) / len(self.recent_losses)

            with open(self.metrics_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        epoch,
                        epoch_stats["train_loss"],
                        smoothed,
                        epoch_stats["kl"],
                        epoch_stats["bce"],
                        w_kl,
                        w_bce,
                        self.optimizer.param_groups[0]["lr"],
                        elapsed,
                    ]
                )

            next_weights = self.loss_balancer.update({"kl": epoch_stats["kl"], "bce": epoch_stats["bce"]})
            self.loss_fn.set_weights(next_weights["kl"], next_weights["bce"])

            self._save_checkpoint("last", epoch)
            # best.pt tracks the best *smoothed training loss* only -
            # validation is never consulted during training (see optim.yaml's
            # early_stop_patience comment) - deliberately, not an oversight.
            if smoothed < self.best_smoothed:
                self.best_smoothed = smoothed
                self.epochs_without_improvement = 0
                self._save_checkpoint("best", epoch)
            else:
                self.epochs_without_improvement += 1

            print(
                f"[{self.cfg['run_name']}] epoch {epoch:3d}  "
                f"loss {epoch_stats['train_loss']:.4f}  smoothed {smoothed:.4f}  "
                f"({elapsed:.1f}s, {self.epochs_without_improvement}/{patience} without improvement)"
            )

            if self.epochs_without_improvement >= patience:
                print(f"[{self.cfg['run_name']}] early stop at epoch {epoch}")
                break

    def _save_checkpoint(self, name: str, epoch: int) -> None:
        # backbone.state_dict() on a peft-wrapped model saves the (frozen)
        # base weights plus the LoRA deltas together — evaluate.py rebuilds
        # the same peft-wrapped architecture and loads this back as-is.
        # Everything past "heads" exists only so last.pt can be resumed from
        # (optimizer/scheduler momentum, early-stop counters, loss-balancer
        # EMA history) - evaluate.py only ever reads backbone/pool/heads.
        state = {
            "epoch": epoch,
            "backbone": self.model.backbone.state_dict(),
            "pool": self.model.pool.state_dict(),
            "heads": self.model.heads.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "loss_balancer": self.loss_balancer.state_dict(),
            "w_kl": self.loss_fn.lambda_kl,
            "w_bce": self.loss_fn.lambda_bce,
            "recent_losses": list(self.recent_losses),
            "best_smoothed": self.best_smoothed,
            "epochs_without_improvement": self.epochs_without_improvement,
        }
        torch.save(state, self.run_dir / "checkpoints" / f"{name}.pt")

    def _load_checkpoint(self, path: Path) -> bool:
        """Returns True on a successful resume, False if path was unusable
        (corrupted/incompatible) - in which case the caller starts fresh
        rather than crash, the same lesson as today's A2 cache corruption:
        a bad file on disk shouldn't be worse than a missing one."""
        try:
            state = torch.load(path, map_location=self.device, weights_only=False)
            self.model.backbone.load_state_dict(state["backbone"])
            self.model.pool.load_state_dict(state["pool"])
            self.model.heads.load_state_dict(state["heads"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.loss_balancer.load_state_dict(state["loss_balancer"])
            self.loss_fn.set_weights(state["w_kl"], state["w_bce"])
            self.recent_losses = deque(state["recent_losses"], maxlen=self.cfg["early_stop_smoothing"])
            self.best_smoothed = state["best_smoothed"]
            self.epochs_without_improvement = state["epochs_without_improvement"]
            self.start_epoch = state["epoch"] + 1
        except Exception as e:
            print(f"[{self.cfg['run_name']}] WARNING: could not resume from {path} ({e!r}) - starting fresh")
            return False

        print(f"[{self.cfg['run_name']}] resumed from {path} - continuing at epoch {self.start_epoch}")
        return True
