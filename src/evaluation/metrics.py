"""The metrics reported per condition (protocole §7): macro AUC, macro
average precision, mean KL to the true distribution, and mean absolute
error to the *reference condition's* prediction on the same images (the
paired comparison, not a signed difference - protocole §7 explains why:
a signed difference can cancel across classes and hide a real
reshuffling of probability mass).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def macro_auc(support: np.ndarray, prob: np.ndarray) -> float:
    scores = []
    for c in range(support.shape[1]):
        col = support[:, c]
        if col.min() == col.max():
            continue  # AUC undefined when a class has only one outcome in this fold
        scores.append(roc_auc_score(col, prob[:, c]))
    return float(np.mean(scores)) if scores else float("nan")


def macro_average_precision(support: np.ndarray, prob: np.ndarray) -> float:
    scores = []
    for c in range(support.shape[1]):
        col = support[:, c]
        if col.max() == 0:
            continue
        scores.append(average_precision_score(col, prob[:, c]))
    return float(np.mean(scores)) if scores else float("nan")


def mean_kl(true_dist: np.ndarray, pred_dist: np.ndarray, eps: float = 1e-12) -> float:
    safe_true = np.clip(true_dist, eps, None)
    safe_pred = np.clip(pred_dist, eps, None)
    kl = (true_dist * (np.log(safe_true) - np.log(safe_pred))).sum(axis=1)
    return float(kl.mean())


def mean_absolute_error_to_reference(pred_dist: np.ndarray, reference_pred_dist: np.ndarray) -> float:
    return float(np.abs(pred_dist - reference_pred_dist).mean(axis=1).mean())


def bootstrap_delta_auc(
    support: np.ndarray,
    prob_condition: np.ndarray,
    prob_reference: np.ndarray,
    families: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict:
    """protocole §7: variance for the inverse-variance fold combination is
    estimated by bootstrap, resampling by family so images that weren't
    separated by the fold split aren't treated as independent."""
    point = macro_auc(support, prob_condition) - macro_auc(support, prob_reference)

    unique_families = np.unique(families)
    deltas = []
    for _ in range(n_bootstrap):
        drawn = rng.choice(unique_families, size=len(unique_families), replace=True)
        idx = np.concatenate([np.where(families == f)[0] for f in drawn])
        d = macro_auc(support[idx], prob_condition[idx]) - macro_auc(support[idx], prob_reference[idx])
        if not np.isnan(d):
            deltas.append(d)

    variance = float(np.var(deltas)) if deltas else float("nan")
    return {"delta_auc": point, "delta_auc_variance": variance, "n_bootstrap_valid": len(deltas)}
