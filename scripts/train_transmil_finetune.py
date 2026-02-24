#!/usr/bin/env python3
"""
TransMIL Fine-Tuning Script for TCGA Ovarian Cancer Platinum Sensitivity Prediction.

This script fine-tunes a TransMIL (Transformer-based Multiple Instance Learning) head
on top of frozen Path Foundation embeddings (384-dim) from the TCGA ovarian cancer
dataset. The goal is slide-level binary classification: predicting platinum sensitivity
(0 = non-responder, 1 = responder).

Key features:
  - **Patient-level** stratified k-fold cross-validation (prevents data leakage
    when multiple slides come from the same patient)
  - Class-weighted focal loss for imbalanced data
  - Cosine annealing with warm restarts
  - Per-fold metrics: AUC-ROC, PR-AUC, accuracy, sensitivity, specificity
  - Calibration curve plotting
  - Training curve plots (loss and AUC over epochs)
  - Best model checkpoint saving
  - Full results JSON logging

IMPORTANT — metric context:
  - The **cross-validation AUC** is the honest out-of-sample metric.
  - The AUC from optimize_threshold.py (which re-runs the best model on ALL
    slides) is a *resubstitution AUC* — useful for threshold calibration but
    NOT a valid performance estimate.

Usage:
    python scripts/train_transmil_finetune.py \\
        --embeddings_dir data/tcga_full/embeddings \\
        --labels_file data/tcga_full/labels.csv \\
        --output_dir results/transmil_finetune

    For a quick smoke test (fewer epochs, 2 folds):
    python scripts/train_transmil_finetune.py \\
        --embeddings_dir data/tcga_full/embeddings \\
        --labels_file data/tcga_full/labels.csv \\
        --output_dir results/transmil_finetune \\
        --n_folds 2 --epochs 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold

# ---------------------------------------------------------------------------
# Resolve imports -- works from repo root or scripts/ directory
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "models"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from transmil import TransMIL  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class FinetuneConfig:
    """All hyperparameters for a single fine-tuning run."""
    # Data
    input_dim: int = 384
    # Model
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.25
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    epochs: int = 100
    patience: int = 20
    # Cross-validation
    n_folds: int = 5
    # Patient-level splitting
    patient_level_cv: bool = True   # <-- NEW: default on
    # Balance the per-epoch training stream by oversampling minority class
    balance_train_epoch: bool = True
    # Minority class feature-space augmentation (applied on top of balanced sampling)
    augment_minority: bool = False
    minority_noise_std: float = 0.015
    minority_feature_dropout: float = 0.05
    minority_mixup_prob: float = 0.25
    minority_mixup_alpha: float = 0.4
    # Cap bag size (patch count) to avoid OOM and bound runtime
    max_train_patches: int = 1024
    max_eval_patches: int = 1024
    # Stratify split targets on label + specimen type (TS/BS) when feasible
    stratify_by_specimen: bool = True
    # Use one train/validation split instead of k-fold CV
    single_split: bool = False
    # Validation fraction used when single_split=True
    val_frac: float = 0.2
    # Hard runtime budget in minutes per fold/split (<=0 disables)
    max_minutes: float = 180.0
    # Misc
    seed: int = 42


# ---------------------------------------------------------------------------
# Loss: Focal Loss for class-imbalanced binary classification
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal loss (Lin et al., 2017) with optional per-sample weighting.

    Reduces the relative loss for well-classified examples and focuses
    training on hard negatives, which is critical for our heavily-skewed
    responder/non-responder split (~85%/15%).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        neg_class_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce

        if neg_class_weight is not None:
            # Apply higher weight to NEGATIVE class samples (minority)
            # to correct for class imbalance.
            weight = torch.where(target == 1, 1.0, neg_class_weight)
            loss = loss * weight

        return loss.mean()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def load_labels(labels_path: Path) -> pd.DataFrame:
    """
    Load the labels CSV.

    Expected columns: slide_id, label (0/1).
    May also contain: patient_id, platinum_status.
    """
    df = pd.read_csv(labels_path)

    required = {"slide_id", "label"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"Labels CSV must contain columns {required}, found {set(df.columns)}"
        )

    return df


def extract_patient_id(slide_id: str) -> str:
    """
    Extract patient ID from a TCGA slide ID.

    Examples:
        TCGA-04-1331-01A-01-BS1.uuid  ->  TCGA-04-1331
        TCGA-04-1331                   ->  TCGA-04-1331

    Falls back to the full slide_id if parsing fails.
    """
    parts = slide_id.split("-")
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])
    return slide_id


def extract_specimen_type(slide_id: str) -> str:
    """
    Extract specimen type token from slide ID.

    Typical TCGA slide IDs contain "-TS1" or "-BS1". If neither exists,
    return "OTHER".
    """
    if "-TS" in slide_id:
        return "TS"
    if "-BS" in slide_id:
        return "BS"
    return "OTHER"


def _uniform_subsample_indices(total: int, target: int) -> np.ndarray:
    """Deterministically pick approximately-uniform indices from [0, total)."""
    if target <= 0 or total <= target:
        return np.arange(total, dtype=np.int64)
    return np.linspace(0, total - 1, num=target, dtype=np.int64)


def load_slide_embeddings(
    embeddings_dir: Path, slide_ids: List[str]
) -> Dict[str, np.ndarray]:
    """
    Load .npy embeddings for the given slide IDs.

    Returns a dict mapping slide_id -> (n_patches, embed_dim) array.
    Slides whose .npy file is missing are silently skipped.
    """
    data: Dict[str, np.ndarray] = {}
    for sid in slide_ids:
        npy_path = embeddings_dir / f"{sid}.npy"
        if npy_path.exists():
            data[sid] = np.load(npy_path)
    return data


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: TransMIL,
    data: Dict[str, np.ndarray],
    labels: Dict[str, int],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    neg_class_weight: torch.Tensor,
    *,
    max_train_patches: int,
    balance_train_epoch: bool,
    augment_minority: bool,
    minority_label: Optional[int],
    minority_ids: List[str],
    minority_noise_std: float,
    minority_feature_dropout: float,
    minority_mixup_prob: float,
    minority_mixup_alpha: float,
    rng: np.random.RandomState,
) -> float:
    """Run a single training epoch over all slides (stochastic / per-slide)."""
    model.train()
    all_ids = list(data.keys())
    if len(all_ids) == 0:
        return 0.0

    if balance_train_epoch:
        pos_ids = [sid for sid in all_ids if labels.get(sid, 0) == 1]
        neg_ids = [sid for sid in all_ids if labels.get(sid, 0) == 0]

        if pos_ids and neg_ids:
            n_target = max(len(pos_ids), len(neg_ids))
            pos_draw = rng.choice(pos_ids, size=n_target, replace=(len(pos_ids) < n_target))
            neg_draw = rng.choice(neg_ids, size=n_target, replace=(len(neg_ids) < n_target))
            slide_ids = np.concatenate([pos_draw, neg_draw]).tolist()
        else:
            slide_ids = all_ids
    else:
        slide_ids = all_ids

    rng.shuffle(slide_ids)

    total_loss = 0.0
    for sid in slide_ids:
        optimizer.zero_grad()

        emb_np = data[sid]
        if max_train_patches > 0 and emb_np.shape[0] > max_train_patches:
            idx = rng.choice(emb_np.shape[0], size=max_train_patches, replace=False)
            emb_np = emb_np[idx]
        emb_np = emb_np.astype(np.float32, copy=False)

        # Minority-class augmentation in feature space.
        # This augments only the underrepresented class to reduce bias.
        if (
            augment_minority
            and minority_label is not None
            and labels.get(sid, 0) == minority_label
            and emb_np.shape[0] > 0
        ):
            if minority_noise_std > 0:
                emb_np = emb_np + rng.normal(
                    0.0, minority_noise_std, size=emb_np.shape
                ).astype(np.float32)

            if minority_feature_dropout > 0:
                keep_prob = max(0.0, 1.0 - minority_feature_dropout)
                mask = (rng.rand(*emb_np.shape) < keep_prob).astype(np.float32)
                emb_np = emb_np * mask

            if minority_mixup_prob > 0 and minority_ids and rng.rand() < minority_mixup_prob:
                sid2 = str(rng.choice(minority_ids))
                emb2 = data[sid2]
                n = emb_np.shape[0]
                if emb2.shape[0] >= n:
                    idx2 = rng.choice(emb2.shape[0], size=n, replace=False)
                else:
                    idx2 = rng.choice(emb2.shape[0], size=n, replace=True)
                emb2 = emb2[idx2].astype(np.float32, copy=False)
                lam = float(rng.beta(minority_mixup_alpha, minority_mixup_alpha))
                emb_np = lam * emb_np + (1.0 - lam) * emb2

        emb = torch.tensor(emb_np, dtype=torch.float32).to(device)
        target = torch.tensor([labels[sid]], dtype=torch.float32).to(device)

        pred = model(emb)
        loss = criterion(pred.view(-1), target, neg_class_weight=neg_class_weight)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(slide_ids), 1)


@torch.no_grad()
def evaluate(
    model: TransMIL,
    data: Dict[str, np.ndarray],
    labels: Dict[str, int],
    device: torch.device,
    *,
    max_eval_patches: int,
) -> Dict[str, float]:
    """
    Evaluate the model on a set of slides.

    Returns a dict with keys: auc, pr_auc, accuracy, sensitivity, specificity, loss.
    """
    model.eval()

    preds: List[float] = []
    true: List[int] = []

    for sid, emb_np in data.items():
        if max_eval_patches > 0 and emb_np.shape[0] > max_eval_patches:
            emb_np = emb_np[_uniform_subsample_indices(emb_np.shape[0], max_eval_patches)]
        emb = torch.tensor(emb_np, dtype=torch.float32).to(device)
        pred = model(emb)
        preds.append(pred.item())
        true.append(labels[sid])

    preds_arr = np.array(preds)
    true_arr = np.array(true)
    pred_binary = (preds_arr >= 0.5).astype(int)

    # AUC-ROC (guard against single-class folds)
    try:
        auc = roc_auc_score(true_arr, preds_arr)
    except ValueError:
        auc = 0.5

    # PR-AUC (average precision) — better metric for imbalanced data
    try:
        pr_auc = average_precision_score(true_arr, preds_arr)
    except ValueError:
        pr_auc = 0.0

    acc = accuracy_score(true_arr, pred_binary)

    # Sensitivity (recall of positive class) and specificity
    tn, fp, fn, tp = confusion_matrix(
        true_arr, pred_binary, labels=[0, 1]
    ).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # BCE loss for monitoring
    loss = F.binary_cross_entropy(
        torch.tensor(preds_arr, dtype=torch.float32),
        torch.tensor(true_arr, dtype=torch.float32),
    ).item()

    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "loss": loss,
    }


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------
def plot_training_curves(
    history: Dict[str, List[float]],
    fold: int,
    output_dir: Path,
) -> None:
    """Save loss and AUC curves for a single fold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # -- Loss --
    axes[0].plot(epochs, history["train_loss"], label="Train", color="#2563eb", lw=2)
    axes[0].plot(epochs, history["val_loss"], label="Val", color="#dc2626", lw=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Fold {fold + 1} -- Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # -- ROC-AUC --
    axes[1].plot(epochs, history["val_auc"], label="Val ROC-AUC", color="#059669", lw=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_title(f"Fold {fold + 1} -- Validation ROC-AUC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # -- PR-AUC --
    axes[2].plot(epochs, history["val_pr_auc"], label="Val PR-AUC", color="#ea580c", lw=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("PR-AUC")
    axes[2].set_title(f"Fold {fold + 1} -- Validation PR-AUC")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"training_curves_fold{fold + 1}.png", dpi=150)
    plt.close()


def plot_summary(fold_metrics: List[Dict], output_dir: Path) -> None:
    """Bar chart summarising per-fold AUC, PR-AUC, accuracy, sensitivity, specificity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(fold_metrics)
    x = np.arange(n)
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(["auc", "pr_auc", "accuracy", "sensitivity", "specificity"]):
        vals = [fm.get(metric, 0) for fm in fold_metrics]
        ax.bar(x + i * width, vals, width, label=metric.replace("_", " ").upper())

    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_title("Per-Fold Metrics Summary (Patient-Level CV)")
    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels([f"Fold {i + 1}" for i in range(n)])
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "summary_metrics.png", dpi=150)
    plt.close()


def plot_calibration_curve(
    all_true: np.ndarray,
    all_preds: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Plot a calibration curve (reliability diagram) from pooled CV predictions.

    A well-calibrated model should have its predicted probabilities closely
    match the observed event frequencies.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    # Compute calibration curve with 10 bins
    fraction_of_positives, mean_predicted_value = calibration_curve(
        all_true, all_preds, n_bins=10, strategy="uniform"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: calibration curve
    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax1.plot(mean_predicted_value, fraction_of_positives, "o-",
             color="#2563eb", lw=2, label="TransMIL")
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title("Calibration Curve (Reliability Diagram)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)

    # Right: histogram of predicted probabilities
    ax2.hist(all_preds[all_true == 1], bins=20, alpha=0.6, color="#2563eb",
             label=f"Positive (n={int(all_true.sum())})")
    ax2.hist(all_preds[all_true == 0], bins=20, alpha=0.6, color="#dc2626",
             label=f"Negative (n={int(len(all_true) - all_true.sum())})")
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "calibration_curve.png", dpi=150)
    plt.close()
    logger.info("Saved calibration curve -> %s", output_dir / "calibration_curve.png")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def run_fold(
    fold: int,
    train_ids: List[str],
    val_ids: List[str],
    all_data: Dict[str, np.ndarray],
    all_labels: Dict[str, int],
    config: FinetuneConfig,
    device: torch.device,
    output_dir: Path,
) -> Tuple[Dict[str, float], Dict[str, List[float]], np.ndarray, np.ndarray]:
    """
    Train and evaluate a single fold.

    Returns (best_metrics, history, val_true, val_preds).
    """
    logger.info(
        "Fold %d/%d  --  train=%d  val=%d",
        fold + 1, config.n_folds, len(train_ids), len(val_ids),
    )

    train_data = {s: all_data[s] for s in train_ids if s in all_data}
    val_data = {s: all_data[s] for s in val_ids if s in all_data}

    # Compute class weight for the training split.
    # neg_class_weight_val = n_pos / n_neg
    #   -> Applied to NEGATIVE class samples in FocalLoss.forward()
    #   -> Upweights the minority (negative/resistant) class
    #   -> Example: 110 pos / 20 neg => neg_class_weight = 5.5
    #      meaning each negative sample contributes 5.5x more to the loss
    train_labels_list = [all_labels[s] for s in train_data]
    n_pos = sum(train_labels_list)
    n_neg = len(train_labels_list) - n_pos
    neg_class_weight_val = n_pos / max(n_neg, 1)
    neg_class_weight = torch.tensor([neg_class_weight_val], dtype=torch.float32).to(device)
    logger.info(
        "  Class distribution: pos=%d neg=%d  neg_class_weight=%.2f",
        n_pos, n_neg, neg_class_weight_val,
    )
    minority_label = None
    if len(train_labels_list) > 0:
        minority_label = 1 if n_pos < n_neg else 0
    minority_ids = [sid for sid in train_data if all_labels[sid] == minority_label] if minority_label is not None else []
    logger.info(
        "  Minority augmentation: %s | minority_label=%s | minority_slides=%d",
        "on" if config.augment_minority else "off",
        str(minority_label) if minority_label is not None else "NA",
        len(minority_ids),
    )
    logger.info(
        "  Patch caps: train_max=%d eval_max=%d | balanced_epoch_sampling=%s",
        config.max_train_patches,
        config.max_eval_patches,
        "on" if config.balance_train_epoch else "off",
    )

    # Build model
    model = TransMIL(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_classes=1,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    best_auc = 0.0
    best_metrics: Dict[str, float] = {}
    best_state = None
    best_val_preds = np.array([])
    best_val_true = np.array([])
    patience_counter = 0

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
        "val_pr_auc": [],
        "val_accuracy": [],
        "val_sensitivity": [],
        "val_specificity": [],
        "lr": [],
    }

    rng = np.random.RandomState(config.seed + fold)
    t0_fold = time.time()
    max_seconds = float(config.max_minutes) * 60.0 if float(config.max_minutes) > 0 else 0.0

    for epoch in range(config.epochs):
        if max_seconds > 0 and (time.time() - t0_fold) >= max_seconds:
            logger.info(
                "  Time budget reached (%.1f min). Stopping fold at epoch %d.",
                float(config.max_minutes),
                epoch + 1,
            )
            break

        t0 = time.time()

        train_loss = train_one_epoch(
            model,
            train_data,
            all_labels,
            device,
            optimizer,
            criterion,
            neg_class_weight,
            max_train_patches=config.max_train_patches,
            balance_train_epoch=config.balance_train_epoch,
            augment_minority=config.augment_minority,
            minority_label=minority_label,
            minority_ids=minority_ids,
            minority_noise_std=config.minority_noise_std,
            minority_feature_dropout=config.minority_feature_dropout,
            minority_mixup_prob=config.minority_mixup_prob,
            minority_mixup_alpha=config.minority_mixup_alpha,
            rng=rng,
        )
        val_metrics = evaluate(
            model,
            val_data,
            all_labels,
            device,
            max_eval_patches=config.max_eval_patches,
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_pr_auc"].append(val_metrics["pr_auc"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_sensitivity"].append(val_metrics["sensitivity"])
        history["val_specificity"].append(val_metrics["specificity"])
        history["lr"].append(current_lr)

        elapsed = time.time() - t0
        logger.info(
            "  Epoch %3d | train_loss=%.4f | val_loss=%.4f | val_auc=%.4f | val_pr_auc=%.4f | val_acc=%.4f | "
            "sens=%.4f | spec=%.4f | lr=%.2e | %.1fs",
            epoch + 1,
            train_loss,
            val_metrics["loss"],
            val_metrics["auc"],
            val_metrics["pr_auc"],
            val_metrics["accuracy"],
            val_metrics["sensitivity"],
            val_metrics["specificity"],
            current_lr,
            elapsed,
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_metrics = val_metrics.copy()
            best_metrics["best_epoch"] = epoch + 1
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
            logger.info("    -> new best AUC: %.4f", best_auc)

            # Capture predictions at best epoch for calibration
            model.eval()
            _preds, _true = [], []
            with torch.no_grad():
                for sid, emb_np in val_data.items():
                    emb = torch.tensor(emb_np, dtype=torch.float32).to(device)
                    pred = model(emb)
                    _preds.append(pred.item())
                    _true.append(all_labels[sid])
            best_val_preds = np.array(_preds)
            best_val_true = np.array(_true)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info("  Early stopping at epoch %d", epoch + 1)
                break

    # Save best model for this fold
    if best_state is not None:
        ckpt_path = output_dir / f"best_model_fold{fold + 1}.pt"
        torch.save(
            {
                "model_state_dict": best_state,
                "config": asdict(config),
                "fold": fold,
                "best_metrics": best_metrics,
            },
            ckpt_path,
        )
        logger.info("  Saved checkpoint -> %s", ckpt_path)

    # Plot curves
    try:
        plot_training_curves(history, fold, output_dir)
    except Exception as exc:
        logger.warning("Could not plot training curves: %s", exc)

    return best_metrics, history, best_val_true, best_val_preds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune TransMIL on TCGA ovarian cancer embeddings."
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        required=True,
        help="Directory with per-slide .npy embedding files.",
    )
    parser.add_argument(
        "--labels_file",
        type=str,
        required=True,
        help="CSV with columns: slide_id, label (0/1). Optionally: patient_id.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/transmil_finetune",
        help="Where to write checkpoints, plots, and results JSON.",
    )
    # Hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--single_split",
        action="store_true",
        help="Use one train/validation split instead of k-fold CV.",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.2,
        help="Validation fraction when --single_split is enabled.",
    )
    parser.add_argument(
        "--max_minutes",
        type=float,
        default=180.0,
        help="Hard runtime budget per fold/split in minutes (<=0 disables).",
    )
    parser.add_argument(
        "--max_train_patches",
        type=int,
        default=1024,
        help="Max patches per slide during training (0 disables subsampling).",
    )
    parser.add_argument(
        "--max_eval_patches",
        type=int,
        default=1024,
        help="Max patches per slide during validation (0 disables subsampling).",
    )
    parser.add_argument(
        "--no_balance_train_epoch",
        action="store_true",
        help="Disable class-balanced per-epoch sampling.",
    )
    parser.add_argument(
        "--augment_minority",
        action="store_true",
        help="Enable minority-class feature-space augmentation.",
    )
    parser.add_argument(
        "--minority_noise_std",
        type=float,
        default=0.015,
        help="Gaussian noise std for minority-class embeddings.",
    )
    parser.add_argument(
        "--minority_feature_dropout",
        type=float,
        default=0.05,
        help="Feature dropout rate for minority-class embeddings.",
    )
    parser.add_argument(
        "--minority_mixup_prob",
        type=float,
        default=0.25,
        help="Probability of same-class mixup on minority-class samples.",
    )
    parser.add_argument(
        "--minority_mixup_alpha",
        type=float,
        default=0.4,
        help="Beta(alpha, alpha) parameter for minority-class mixup.",
    )
    parser.add_argument(
        "--no_stratify_specimen",
        action="store_true",
        help="Stratify on label only instead of label+specimen type.",
    )
    parser.add_argument(
        "--no_patient_level_cv",
        action="store_true",
        help="Disable patient-level CV (NOT recommended — risks data leakage).",
    )

    args = parser.parse_args()

    # Build config
    config = FinetuneConfig(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        n_folds=args.n_folds,
        seed=args.seed,
        patient_level_cv=not args.no_patient_level_cv,
        balance_train_epoch=not args.no_balance_train_epoch,
        augment_minority=args.augment_minority,
        minority_noise_std=args.minority_noise_std,
        minority_feature_dropout=args.minority_feature_dropout,
        minority_mixup_prob=args.minority_mixup_prob,
        minority_mixup_alpha=args.minority_mixup_alpha,
        max_train_patches=args.max_train_patches,
        max_eval_patches=args.max_eval_patches,
        stratify_by_specimen=not args.no_stratify_specimen,
        single_split=args.single_split,
        val_frac=args.val_frac,
        max_minutes=args.max_minutes,
    )

    # Seed everything
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    labels_df = load_labels(Path(args.labels_file))
    logger.info("Loaded %d label entries from %s", len(labels_df), args.labels_file)

    embeddings_dir = Path(args.embeddings_dir)
    all_slide_ids = labels_df["slide_id"].tolist()
    all_data = load_slide_embeddings(embeddings_dir, all_slide_ids)

    # Keep only slides that have both labels and embeddings
    available_ids = sorted(all_data.keys())
    labels_map: Dict[str, int] = dict(
        zip(labels_df["slide_id"], labels_df["label"].astype(int))
    )
    available_ids = [s for s in available_ids if s in labels_map]
    available_labels = np.array([labels_map[s] for s in available_ids])
    available_specimen = np.array([extract_specimen_type(s) for s in available_ids])

    n_pos = int(available_labels.sum())
    n_neg = len(available_labels) - n_pos
    logger.info(
        "Available slides: %d  (pos=%d [%.1f%%], neg=%d [%.1f%%])",
        len(available_ids),
        n_pos,
        100 * n_pos / len(available_ids),
        n_neg,
        100 * n_neg / len(available_ids),
    )
    specimen_counts = pd.Series(available_specimen).value_counts().to_dict()
    logger.info("Specimen distribution: %s", specimen_counts)

    if len(available_ids) == 0:
        logger.error("No slides found with both embeddings and labels. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Resolve patient IDs for patient-level splitting
    # ------------------------------------------------------------------
    if "patient_id" in labels_df.columns:
        patient_map = dict(zip(labels_df["slide_id"], labels_df["patient_id"]))
        logger.info("Using patient_id column from labels CSV.")
    else:
        # Derive from TCGA slide_id naming convention
        patient_map = {sid: extract_patient_id(sid) for sid in available_ids}
        logger.info("Derived patient_id from slide_id naming convention.")

    patient_groups = np.array([patient_map.get(s, s) for s in available_ids])
    n_unique_patients = len(set(patient_groups))
    n_multi_slide = len(available_ids) - n_unique_patients
    logger.info(
        "Patients: %d unique (from %d slides; %d slides share a patient_id)",
        n_unique_patients,
        len(available_ids),
        n_multi_slide,
    )

    if n_multi_slide > 0 and config.patient_level_cv:
        logger.warning(
            "⚠ %d slides share patient IDs — patient-level CV is CRITICAL "
            "to prevent data leakage. Using StratifiedGroupKFold.",
            n_multi_slide,
        )

    # Check embedding dimension
    sample = all_data[available_ids[0]]
    actual_dim = sample.shape[1]
    if actual_dim != config.input_dim:
        logger.warning(
            "Embedding dim=%d differs from config.input_dim=%d; overriding.",
            actual_dim,
            config.input_dim,
        )
        config.input_dim = actual_dim

    # ------------------------------------------------------------------
    # Cross-validation (patient-level or slide-level)
    # ------------------------------------------------------------------
    if config.single_split:
        if not (0.05 <= float(config.val_frac) <= 0.5):
            logger.warning(
                "val_frac=%.3f is outside recommended [0.05, 0.5]; clamping.",
                float(config.val_frac),
            )
            config.val_frac = max(0.05, min(0.5, float(config.val_frac)))
        target_splits = max(2, int(round(1.0 / float(config.val_frac))))
    else:
        target_splits = config.n_folds

    split_targets = available_labels
    cv_strat_target = "label"
    if config.stratify_by_specimen:
        combo = np.array([f"{int(y)}|{s}" for y, s in zip(available_labels, available_specimen)])
        combo_counts = pd.Series(combo).value_counts()
        if combo_counts.size > 1 and int(combo_counts.min()) >= target_splits:
            split_targets = combo
            cv_strat_target = "label+specimen"
        else:
            logger.warning(
                "Cannot stratify by label+specimen for %d splits (min combo count=%d). Falling back to label-only.",
                target_splits,
                int(combo_counts.min()) if len(combo_counts) else 0,
            )

    if config.single_split:
        n_total = len(available_ids)
        if n_total < 2:
            raise ValueError("Need at least 2 slides for a train/validation split.")

        if config.patient_level_cv:
            if n_unique_patients < 2:
                raise ValueError(
                    "Patient-level split requested but fewer than 2 unique patients are available."
                )

            # Prefer stratified grouped split when enough groups exist; otherwise still
            # keep strict patient isolation via GroupShuffleSplit.
            if n_unique_patients >= target_splits:
                splitter = StratifiedGroupKFold(
                    n_splits=target_splits, shuffle=True, random_state=config.seed
                )
                split_pairs = list(splitter.split(available_ids, split_targets, groups=patient_groups))
                target_val = int(round(float(config.val_frac) * n_total))
                best_i = 0
                best_gap = None
                for i, (_tr, va) in enumerate(split_pairs):
                    gap = abs(len(va) - target_val)
                    if best_gap is None or gap < best_gap:
                        best_gap = gap
                        best_i = i
                split_iter = [split_pairs[best_i]]
                split_kind = "StratifiedGroupKFold"
                split_level = "patient-level"
                logger.info(
                    "Using %s (%s, strat=%s) for single split: val_frac=%.3f (picked split %d/%d with val=%d/%d).",
                    split_kind,
                    split_level,
                    cv_strat_target,
                    float(config.val_frac),
                    best_i + 1,
                    len(split_pairs),
                    len(split_pairs[best_i][1]),
                    n_total,
                )
            else:
                splitter = GroupShuffleSplit(
                    n_splits=1, test_size=float(config.val_frac), random_state=config.seed
                )
                split_iter = list(splitter.split(available_ids, groups=patient_groups))
                split_kind = "GroupShuffleSplit"
                split_level = "patient-level"
                logger.warning(
                    "Using GroupShuffleSplit for single split because unique patients (%d) < target_splits (%d). "
                    "Patient isolation is preserved, but stratification may be weaker.",
                    n_unique_patients,
                    target_splits,
                )
                logger.info(
                    "Using %s (%s, strat=%s) for single split: val_frac=%.3f (val=%d/%d).",
                    split_kind,
                    split_level,
                    cv_strat_target,
                    float(config.val_frac),
                    len(split_iter[0][1]),
                    n_total,
                )
        else:
            splitter = StratifiedKFold(
                n_splits=target_splits, shuffle=True, random_state=config.seed
            )
            split_pairs = list(splitter.split(available_ids, split_targets))
            target_val = int(round(float(config.val_frac) * n_total))
            best_i = 0
            best_gap = None
            for i, (_tr, va) in enumerate(split_pairs):
                gap = abs(len(va) - target_val)
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best_i = i
            split_iter = [split_pairs[best_i]]
            split_kind = "StratifiedKFold"
            split_level = "slide-level"
            logger.info(
                "Using %s (%s, strat=%s) for single split: val_frac=%.3f (picked split %d/%d with val=%d/%d).",
                split_kind,
                split_level,
                cv_strat_target,
                float(config.val_frac),
                best_i + 1,
                len(split_pairs),
                len(split_pairs[best_i][1]),
                n_total,
            )
    else:
        if config.patient_level_cv:
            if n_unique_patients < target_splits:
                raise ValueError(
                    f"Patient-level split requires at least n_folds unique patients. "
                    f"Got {n_unique_patients} unique patients for n_folds={target_splits}."
                )
            splitter = StratifiedGroupKFold(
                n_splits=target_splits, shuffle=True, random_state=config.seed
            )
            split_iter = splitter.split(available_ids, split_targets, groups=patient_groups)
            split_kind = "StratifiedGroupKFold"
            split_level = "patient-level"
        else:
            splitter = StratifiedKFold(
                n_splits=target_splits, shuffle=True, random_state=config.seed
            )
            split_iter = splitter.split(available_ids, split_targets)
            split_kind = "StratifiedKFold"
            split_level = "slide-level"

        logger.info(
            "Using %s (%s, %d folds, strat=%s)",
            split_kind,
            split_level,
            target_splits,
            cv_strat_target,
        )

    available_ids_arr = np.array(available_ids)

    fold_results: List[Dict] = []
    fold_histories: List[Dict] = []
    all_cv_true: List[np.ndarray] = []
    all_cv_preds: List[np.ndarray] = []

    if config.single_split:
        logger.info(
            "\n" + "=" * 70 + "\n"
            "  TransMIL Fine-Tuning  --  single split (%s, strat=%s)\n" + "=" * 70,
            split_level,
            cv_strat_target,
        )
    else:
        logger.info(
            "\n" + "=" * 70 + "\n"
            "  TransMIL Fine-Tuning  --  %d-fold CV (%s, strat=%s)\n" + "=" * 70,
            target_splits,
            split_level,
            cv_strat_target,
        )

    for fold, (train_idx, val_idx) in enumerate(split_iter):
        train_ids = available_ids_arr[train_idx].tolist()
        val_ids = available_ids_arr[val_idx].tolist()

        # Verify no patient leakage (debug check)
        if split_level == "patient-level":
            train_patients = set(patient_map.get(s, s) for s in train_ids)
            val_patients = set(patient_map.get(s, s) for s in val_ids)
            leaked = train_patients & val_patients
            if leaked:
                logger.error(
                    "DATA LEAKAGE DETECTED: %d patients in both train and val: %s",
                    len(leaked), list(leaked)[:5],
                )
                raise RuntimeError(f"Patient-level data leakage detected: {leaked}")
            logger.info(
                "  Fold %d: train_patients=%d, val_patients=%d, no leakage ✓",
                fold + 1, len(train_patients), len(val_patients),
            )

        metrics, history, val_true, val_preds = run_fold(
            fold=fold,
            train_ids=train_ids,
            val_ids=val_ids,
            all_data=all_data,
            all_labels=labels_map,
            config=config,
            device=device,
            output_dir=output_dir,
        )
        fold_results.append(metrics)
        fold_histories.append(history)
        all_cv_true.append(val_true)
        all_cv_preds.append(val_preds)

    # ------------------------------------------------------------------
    # Aggregate and report
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info(
        "%s RESULTS (%s)",
        "SINGLE-SPLIT" if config.single_split else "CROSS-VALIDATION",
        split_level,
    )
    logger.info("=" * 70)

    metric_names = ["auc", "pr_auc", "accuracy", "sensitivity", "specificity"]
    agg: Dict[str, Dict[str, float]] = {}

    for m in metric_names:
        vals = [fr[m] for fr in fold_results if m in fr]
        agg[m] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "per_fold": vals,
        }
        logger.info(
            "  %-15s  %.4f +/- %.4f   %s",
            m,
            agg[m]["mean"],
            agg[m]["std"],
            ["%.4f" % v for v in vals],
        )

    # Pick the fold with highest AUC as the "best model"
    best_fold = int(np.argmax([fr.get("auc", 0) for fr in fold_results]))
    best_ckpt = output_dir / f"best_model_fold{best_fold + 1}.pt"
    overall_best = output_dir / "best_model.pt"

    if best_ckpt.exists():
        import shutil
        shutil.copy2(best_ckpt, overall_best)
        logger.info(
            "\nBest fold: %d (AUC=%.4f) -- copied to %s",
            best_fold + 1,
            fold_results[best_fold]["auc"],
            overall_best,
        )

    # Summary plot
    try:
        plot_summary(fold_results, output_dir)
    except Exception as exc:
        logger.warning("Could not plot summary: %s", exc)

    # Calibration curve from pooled CV predictions
    try:
        pooled_true = np.concatenate(all_cv_true)
        pooled_preds = np.concatenate(all_cv_preds)
        if len(pooled_true) > 0:
            plot_calibration_curve(pooled_true, pooled_preds, output_dir)
    except Exception as exc:
        logger.warning("Could not plot calibration curve: %s", exc)

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": asdict(config),
        "device": str(device),
        "n_slides": len(available_ids),
        "n_unique_patients": n_unique_patients,
        "class_distribution": {"positive": n_pos, "negative": n_neg},
        "cv_type": (
            f"single-split ({split_level}, {split_kind})"
            if config.single_split
            else f"{split_level} ({split_kind})"
        ),
        "cv_stratification_target": cv_strat_target,
        "aggregate_metrics": agg,
        "per_fold_metrics": fold_results,
        "per_fold_histories": fold_histories,
        "best_fold": best_fold + 1,
        "best_fold_auc": fold_results[best_fold].get("auc", 0),
        "NOTE": (
            "The metrics above are CROSS-VALIDATED (out-of-sample). "
            "The AUC from optimize_threshold.py is a RESUBSTITUTION AUC "
            "(model evaluated on its own training data) and is NOT a valid "
            "performance estimate."
        ),
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results written to %s", results_path)

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
