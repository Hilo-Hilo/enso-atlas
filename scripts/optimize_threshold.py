#!/usr/bin/env python3
"""
Optimal Threshold Selection for TransMIL Classifier.

Loads a trained TransMIL checkpoint, runs inference on ALL slides
(both train and validation), and computes the optimal decision
threshold using:

  1. Youden's J statistic  (sensitivity + specificity - 1)
  2. F1-optimal threshold

⚠️  IMPORTANT — Metric Context:
    The AUC reported by this script is a **RESUBSTITUTION AUC**:
    it evaluates the model on the same data used (in full or in part)
    for training. This is NOT a valid out-of-sample performance estimate.

    The honest cross-validated AUC comes from train_transmil_finetune.py
    (patient-level 5-fold CV).

    Use this script's threshold for production calibration, but cite
    the CV AUC for performance claims.

Generates:
  - Score distribution histogram (positive vs negative)
  - Threshold vs metrics curve (sensitivity, specificity, F1, Youden J)
  - Summary config JSON with the chosen threshold

Usage (run on DGX Spark where embeddings + model live):

    python scripts/optimize_threshold.py \
        --checkpoint results/transmil_training/best_model.pt \
        --embeddings_dir data/tcga_full/embeddings \
        --labels_file data/tcga_full/matched_labels.csv \
        --output_dir results/threshold_optimization

    # Or with explicit paths:
    python scripts/optimize_threshold.py \
        --checkpoint outputs/training/best_model.pt \
        --embeddings_dir data/embeddings \
        --labels_file data/tcga_full/labels.csv \
        --output_dir results/threshold_optimization
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resolve imports
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "models"))
sys.path.insert(0, str(_REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Model loading (supports both TransMIL and CLAM checkpoints)
# ---------------------------------------------------------------------------

def _detect_architecture(state_dict: dict) -> str:
    """Detect model architecture from state dict keys."""
    keys = set(state_dict.keys())
    if any(k.startswith("fc_in.") for k in keys):
        return "transmil"
    if any(k.startswith("feature_extractor.") for k in keys):
        return "clam"
    if any(k.startswith("encoder.") for k in keys):
        return "clam_legacy"
    return "unknown"


def load_model(checkpoint_path: Path, device):
    """
    Load a model from checkpoint, auto-detecting architecture.

    Returns (model, config_dict, architecture_name).
    """
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        cfg = checkpoint.get("config", {})
    else:
        state_dict = checkpoint
        cfg = {}

    arch = _detect_architecture(state_dict)
    logger.info("Detected architecture: %s", arch)

    if arch == "transmil":
        from transmil import TransMIL
        model = TransMIL(
            input_dim=cfg.get("input_dim", 384),
            hidden_dim=cfg.get("hidden_dim", 512),
            num_classes=cfg.get("num_classes", 1),
            num_heads=cfg.get("num_heads", 8),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.25),
        )
    elif arch in ("clam", "clam_legacy"):
        # Defer to the CLAM loader in mil module
        from enso_atlas.mil.clam import CLAMClassifier
        from enso_atlas.config import MILConfig
        mil_cfg = MILConfig(
            input_dim=cfg.get("input_dim", 384),
            hidden_dim=cfg.get("hidden_dim", 256),
        )
        clf = CLAMClassifier(mil_cfg)
        clf.load(checkpoint_path)
        return clf._model, cfg, arch
    else:
        raise ValueError(
            f"Cannot detect architecture from checkpoint keys: "
            f"{list(state_dict.keys())[:5]}..."
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, cfg, arch


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_labels(labels_path: Path) -> pd.DataFrame:
    """Load labels CSV (slide_id, label columns required)."""
    df = pd.read_csv(labels_path)
    if "slide_id" not in df.columns or "label" not in df.columns:
        raise ValueError("Labels CSV must have 'slide_id' and 'label' columns.")
    return df


def load_embeddings(
    embeddings_dir: Path, slide_ids: List[str]
) -> Dict[str, np.ndarray]:
    """Load .npy embeddings for the listed slide IDs."""
    data: Dict[str, np.ndarray] = {}
    for sid in slide_ids:
        p = embeddings_dir / f"{sid}.npy"
        if p.exists():
            data[sid] = np.load(p)
    return data


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_all(model, data: Dict[str, np.ndarray], device) -> Dict[str, float]:
    """Run inference on all slides. Returns slide_id -> probability."""
    import torch

    model.eval()
    results: Dict[str, float] = {}

    with torch.no_grad():
        for sid, emb_np in data.items():
            emb = torch.tensor(emb_np, dtype=torch.float32).to(device)
            prob, _ = model(emb, return_attention=True)
            results[sid] = prob.item()

    return results


# ---------------------------------------------------------------------------
# Threshold optimization
# ---------------------------------------------------------------------------

def compute_metrics_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> Dict[str, float]:
    """Compute classification metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    youden_j = sensitivity + specificity - 1.0

    return {
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
        "youden_j": youden_j,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def find_optimal_thresholds(
    y_true: np.ndarray, y_prob: np.ndarray, n_points: int = 1000
) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, float]]]:
    """
    Find optimal thresholds using Youden's J and F1.

    Returns:
        (youden_best, f1_best, all_metrics_list)
    """
    thresholds = np.linspace(0.0, 1.0, n_points)
    all_metrics = []

    best_youden = {"youden_j": -999.0}
    best_f1 = {"f1": -999.0}

    for t in thresholds:
        m = compute_metrics_at_threshold(y_true, y_prob, t)
        all_metrics.append(m)

        if m["youden_j"] > best_youden["youden_j"]:
            best_youden = m.copy()
        if m["f1"] > best_f1["f1"]:
            best_f1 = m.copy()

    return best_youden, best_f1, all_metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_score_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    youden_thresh: float,
    f1_thresh: float,
    output_path: Path,
) -> None:
    """Save score distribution histogram for positive vs negative slides."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos_scores = y_prob[y_true == 1]
    neg_scores = y_prob[y_true == 0]

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0, 1, 40)
    ax.hist(pos_scores, bins=bins, alpha=0.6, color="#2563eb", label=f"Responder (n={len(pos_scores)})")
    ax.hist(neg_scores, bins=bins, alpha=0.6, color="#dc2626", label=f"Non-Responder (n={len(neg_scores)})")

    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Default (0.5)")
    ax.axvline(youden_thresh, color="#16a34a", linestyle="-", linewidth=2,
               label=f"Youden J ({youden_thresh:.3f})")
    ax.axvline(f1_thresh, color="#ea580c", linestyle="-", linewidth=2,
               label=f"F1-optimal ({f1_thresh:.3f})")

    ax.set_xlabel("Predicted Probability (Responder)")
    ax.set_ylabel("Count")
    ax.set_title("TransMIL Score Distribution by True Label")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved score distribution -> %s", output_path)


def plot_threshold_vs_metrics(
    all_metrics: List[Dict[str, float]],
    youden_thresh: float,
    f1_thresh: float,
    output_path: Path,
) -> None:
    """Save threshold vs metrics (sensitivity, specificity, F1, Youden J) curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = [m["threshold"] for m in all_metrics]
    sensitivities = [m["sensitivity"] for m in all_metrics]
    specificities = [m["specificity"] for m in all_metrics]
    f1_scores = [m["f1"] for m in all_metrics]
    youden_values = [m["youden_j"] for m in all_metrics]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left panel: sensitivity and specificity
    ax = axes[0]
    ax.plot(thresholds, sensitivities, color="#2563eb", linewidth=2, label="Sensitivity (TPR)")
    ax.plot(thresholds, specificities, color="#dc2626", linewidth=2, label="Specificity (TNR)")
    ax.axvline(youden_thresh, color="#16a34a", linestyle="--", linewidth=1.5,
               label=f"Youden J ({youden_thresh:.3f})")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, label="Default (0.5)")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title("Sensitivity / Specificity vs Threshold")
    ax.legend(loc="center left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([-0.05, 1.05])

    # Right panel: F1 and Youden J
    ax = axes[1]
    ax.plot(thresholds, f1_scores, color="#ea580c", linewidth=2, label="F1 Score")
    ax.plot(thresholds, youden_values, color="#16a34a", linewidth=2, label="Youden J")
    ax.axvline(youden_thresh, color="#16a34a", linestyle="--", linewidth=1.5,
               label=f"Youden J opt ({youden_thresh:.3f})")
    ax.axvline(f1_thresh, color="#ea580c", linestyle="--", linewidth=1.5,
               label=f"F1 opt ({f1_thresh:.3f})")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, label="Default (0.5)")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title("F1 Score / Youden J vs Threshold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved threshold vs metrics -> %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute optimal decision threshold for a trained MIL model."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pt checkpoint file.",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        required=True,
        help="Directory with per-slide .npy embeddings.",
    )
    parser.add_argument(
        "--labels_file",
        type=str,
        required=True,
        help="CSV with slide_id and label columns.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/threshold_optimization",
        help="Output directory for plots and config.",
    )
    args = parser.parse_args()

    import torch

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    # Load model
    model, model_cfg, arch = load_model(Path(args.checkpoint), device)
    logger.info("Model config: %s", model_cfg)

    # Load data
    labels_df = load_labels(Path(args.labels_file))
    labels_map: Dict[str, int] = dict(
        zip(labels_df["slide_id"], labels_df["label"].astype(int))
    )
    all_slide_ids = list(labels_map.keys())

    embeddings = load_embeddings(Path(args.embeddings_dir), all_slide_ids)
    matched_ids = [s for s in all_slide_ids if s in embeddings]
    logger.info(
        "Matched %d / %d slides with embeddings",
        len(matched_ids), len(all_slide_ids),
    )

    if len(matched_ids) == 0:
        logger.error("No slides found with embeddings. Check paths.")
        sys.exit(1)

    matched_data = {s: embeddings[s] for s in matched_ids}

    # Run inference
    logger.info("Running inference on %d slides...", len(matched_ids))
    predictions = predict_all(model, matched_data, device)

    y_true = np.array([labels_map[s] for s in matched_ids])
    y_prob = np.array([predictions[s] for s in matched_ids])

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    logger.info("Class distribution: %d positive, %d negative (%.1f%% positive)",
                n_pos, n_neg, 100.0 * n_pos / len(y_true))

    # Compute AUC (RESUBSTITUTION — see docstring warning)
    from sklearn.metrics import roc_auc_score, average_precision_score
    try:
        auc_val = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_val = 0.5
    try:
        pr_auc_val = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc_val = 0.0
    logger.info("RESUBSTITUTION AUC-ROC: %.4f (full-dataset, NOT out-of-sample)", auc_val)
    logger.info("RESUBSTITUTION PR-AUC:  %.4f (full-dataset, NOT out-of-sample)", pr_auc_val)
    logger.info("For honest out-of-sample AUC, see train_transmil_finetune.py cross-validation results.")

    # Find optimal thresholds
    logger.info("Computing optimal thresholds...")
    youden_best, f1_best, all_metrics = find_optimal_thresholds(y_true, y_prob)

    logger.info(
        "Youden J optimal: threshold=%.4f  sens=%.3f  spec=%.3f  J=%.3f",
        youden_best["threshold"],
        youden_best["sensitivity"],
        youden_best["specificity"],
        youden_best["youden_j"],
    )
    logger.info(
        "F1 optimal:       threshold=%.4f  sens=%.3f  spec=%.3f  F1=%.3f",
        f1_best["threshold"],
        f1_best["sensitivity"],
        f1_best["specificity"],
        f1_best["f1"],
    )

    # Default threshold comparison
    default_metrics = compute_metrics_at_threshold(y_true, y_prob, 0.5)
    logger.info(
        "Default (0.5):    sens=%.3f  spec=%.3f  F1=%.3f",
        default_metrics["sensitivity"],
        default_metrics["specificity"],
        default_metrics["f1"],
    )

    # Generate plots
    logger.info("Generating plots...")
    plot_score_distribution(
        y_true, y_prob,
        youden_best["threshold"], f1_best["threshold"],
        output_dir / "score_distribution.png",
    )
    plot_threshold_vs_metrics(
        all_metrics,
        youden_best["threshold"], f1_best["threshold"],
        output_dir / "threshold_vs_metrics.png",
    )

    # Save per-slide predictions
    preds_df = pd.DataFrame({
        "slide_id": matched_ids,
        "true_label": y_true.tolist(),
        "predicted_prob": [round(float(predictions[s]), 6) for s in matched_ids],
    })
    preds_df.to_csv(output_dir / "predictions.csv", index=False)

    # Save threshold config (this is what gets loaded by the app)
    threshold_config = {
        "model_checkpoint": str(args.checkpoint),
        "architecture": arch,
        "auc_roc_resubstitution": round(auc_val, 4),
        "pr_auc_resubstitution": round(pr_auc_val, 4),
        "WARNING": "AUC values are RESUBSTITUTION (full-dataset). For out-of-sample AUC see train_transmil_finetune.py results.",
        "n_slides": len(matched_ids),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "optimal_threshold_youden": round(youden_best["threshold"], 4),
        "optimal_threshold_f1": round(f1_best["threshold"], 4),
        "recommended_threshold": round(youden_best["threshold"], 4),
        "metrics_at_recommended": {
            "sensitivity": round(youden_best["sensitivity"], 4),
            "specificity": round(youden_best["specificity"], 4),
            "precision": round(youden_best["precision"], 4),
            "f1": round(youden_best["f1"], 4),
            "accuracy": round(youden_best["accuracy"], 4),
            "youden_j": round(youden_best["youden_j"], 4),
        },
        "metrics_at_f1_optimal": {
            "threshold": round(f1_best["threshold"], 4),
            "sensitivity": round(f1_best["sensitivity"], 4),
            "specificity": round(f1_best["specificity"], 4),
            "f1": round(f1_best["f1"], 4),
        },
        "metrics_at_default_0_5": {
            "sensitivity": round(default_metrics["sensitivity"], 4),
            "specificity": round(default_metrics["specificity"], 4),
            "f1": round(default_metrics["f1"], 4),
        },
    }

    config_path = output_dir / "threshold_config.json"
    with open(config_path, "w") as f:
        json.dump(threshold_config, f, indent=2)
    logger.info("Saved threshold config -> %s", config_path)

    # Also save to models/ for the app to find
    app_config_path = _REPO_ROOT / "models" / "threshold_config.json"
    with open(app_config_path, "w") as f:
        json.dump(threshold_config, f, indent=2)
    logger.info("Saved app threshold config -> %s", app_config_path)

    # Summary JSON with full details
    summary = {
        "threshold_config": threshold_config,
        "model_config": model_cfg,
        "youden_optimal": youden_best,
        "f1_optimal": f1_best,
        "default_0_5": default_metrics,
        "score_statistics": {
            "mean": round(float(y_prob.mean()), 4),
            "std": round(float(y_prob.std()), 4),
            "min": round(float(y_prob.min()), 4),
            "max": round(float(y_prob.max()), 4),
            "median": round(float(np.median(y_prob)), 4),
            "positive_mean": round(float(y_prob[y_true == 1].mean()), 4) if n_pos > 0 else None,
            "negative_mean": round(float(y_prob[y_true == 0].mean()), 4) if n_neg > 0 else None,
        },
    }

    summary_path = output_dir / "optimization_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved full summary -> %s", summary_path)

    logger.info("Threshold optimization complete. Outputs in %s", output_dir)
    logger.info(
        "RECOMMENDED THRESHOLD: %.4f  (Youden J = %.3f)",
        threshold_config["recommended_threshold"],
        youden_best["youden_j"],
    )


if __name__ == "__main__":
    main()
