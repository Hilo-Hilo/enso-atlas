#!/usr/bin/env python3
"""
Model Evaluation Script for CLAM on Ovarian Cancer.

Features:
- Held-out test set performance
- Confidence intervals via bootstrapping
- Calibration analysis (reliability diagram)
- Attention visualization on test cases
- ROC curve with confidence band

Usage:
    python scripts/evaluate_model.py --model_path outputs/training/best_model.pt --data_dir data/demo
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def bootstrap_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute metric with bootstrap confidence intervals.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        metric_fn: Function(y_true, y_prob) -> float
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval level (e.g., 0.95 for 95% CI)
        random_seed: Random seed
        
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    np.random.seed(random_seed)
    
    n = len(y_true)
    point_estimate = metric_fn(y_true, y_prob)
    
    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        
        # Skip if only one class in bootstrap sample
        if len(np.unique(y_true_boot)) < 2:
            continue
            
        try:
            estimate = metric_fn(y_true_boot, y_prob_boot)
            bootstrap_estimates.append(estimate)
        except Exception:
            continue
    
    if len(bootstrap_estimates) < 10:
        logger.warning("Too few valid bootstrap samples")
        return point_estimate, point_estimate, point_estimate
    
    bootstrap_estimates = np.array(bootstrap_estimates)
    
    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
    
    return point_estimate, ci_lower, ci_upper


def compute_all_metrics_with_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 1000
) -> Dict:
    """
    Compute all clinical metrics with bootstrap confidence intervals.
    """
    from sklearn.metrics import (
        roc_auc_score, brier_score_loss,
        confusion_matrix, accuracy_score
    )
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    
    results = {}
    
    # Define metric functions
    def auc_fn(y_t, y_p):
        return roc_auc_score(y_t, y_p)
    
    def sensitivity_fn(y_t, y_p):
        y_pred = (y_p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def specificity_fn(y_t, y_p):
        y_pred = (y_p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def ppv_fn(y_t, y_p):
        y_pred = (y_p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    def npv_fn(y_t, y_p):
        y_pred = (y_p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
        return tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    def brier_fn(y_t, y_p):
        return brier_score_loss(y_t, y_p)
    
    def accuracy_fn(y_t, y_p):
        y_pred = (y_p >= threshold).astype(int)
        return accuracy_score(y_t, y_pred)
    
    metrics_to_compute = [
        ("auc_roc", auc_fn),
        ("sensitivity", sensitivity_fn),
        ("specificity", specificity_fn),
        ("ppv", ppv_fn),
        ("npv", npv_fn),
        ("brier_score", brier_fn),
        ("accuracy", accuracy_fn),
    ]
    
    for metric_name, metric_fn in metrics_to_compute:
        point, lower, upper = bootstrap_metric(
            y_true, y_prob, metric_fn, n_bootstrap=n_bootstrap
        )
        results[metric_name] = {
            "value": point,
            "ci_lower": lower,
            "ci_upper": upper,
            "ci_95": f"{point:.3f} [{lower:.3f}, {upper:.3f}]"
        }
    
    # Confusion matrix (no CI needed)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    results["confusion_matrix"] = {
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
    }
    
    return results


def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict:
    """
    Compute calibration metrics and data for reliability diagram.
    
    Returns:
        Dictionary with calibration data
    """
    from sklearn.calibration import calibration_curve
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    
    # Expected Calibration Error (ECE)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += (mask.sum() / len(y_prob)) * abs(bin_acc - bin_conf)
    
    # Maximum Calibration Error (MCE)
    mce = np.max(np.abs(prob_true - prob_pred)) if len(prob_true) > 0 else 0.0
    
    return {
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(mce),
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "n_bins": n_bins
    }


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    n_bootstrap: int = 100
):
    """
    Plot ROC curve with bootstrap confidence band.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Base ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Bootstrap for confidence band
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        
        fpr_boot, tpr_boot, _ = roc_curve(y_true[indices], y_prob[indices])
        tprs.append(np.interp(mean_fpr, fpr_boot, tpr_boot))
        aucs.append(auc(fpr_boot, tpr_boot))
    
    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')
    ax.plot(fpr, tpr, color='blue', lw=2, 
            label=f'ROC (AUC = {roc_auc:.3f})')
    
    # Confidence band
    ax.fill_between(mean_fpr, 
                    np.maximum(mean_tpr - 1.96 * std_tpr, 0),
                    np.minimum(mean_tpr + 1.96 * std_tpr, 1),
                    color='blue', alpha=0.2, label='95% CI')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - CLAM Model', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved ROC curve to {output_path}")


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    n_bins: int = 10
):
    """
    Plot reliability diagram (calibration curve).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    
    # Model calibration
    ax.plot(prob_pred, prob_true, 'o-', color='blue', label='CLAM', markersize=8)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Calibration Curve (Reliability Diagram)', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved calibration curve to {output_path}")


def plot_attention_visualization(
    embeddings: np.ndarray,
    attention_weights: np.ndarray,
    coords: Optional[np.ndarray],
    slide_id: str,
    label: int,
    prediction: float,
    output_path: Path
):
    """
    Visualize attention weights for a slide.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Attention distribution
    ax1 = axes[0]
    sorted_indices = np.argsort(attention_weights)[::-1]
    ax1.bar(range(len(attention_weights)), attention_weights[sorted_indices], 
            color='steelblue', alpha=0.7)
    ax1.set_xlabel('Patch (sorted by attention)', fontsize=12)
    ax1.set_ylabel('Attention Weight', fontsize=12)
    ax1.set_title('Attention Distribution', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Right: Spatial heatmap (if coords available)
    ax2 = axes[1]
    if coords is not None and len(coords) == len(attention_weights):
        norm = Normalize(vmin=attention_weights.min(), vmax=attention_weights.max())
        
        scatter = ax2.scatter(
            coords[:, 0], coords[:, 1],
            c=attention_weights,
            cmap='jet',
            s=50,
            alpha=0.8
        )
        plt.colorbar(scatter, ax=ax2, label='Attention')
        ax2.set_xlabel('X coordinate', fontsize=12)
        ax2.set_ylabel('Y coordinate', fontsize=12)
        ax2.set_title('Spatial Attention Map', fontsize=14)
        ax2.invert_yaxis()  # Match image coordinate system
    else:
        # Show histogram instead
        ax2.hist(attention_weights, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Attention Weight', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Attention Weight Distribution', fontsize=14)
    
    # Overall title
    label_str = "Responder" if label == 1 else "Non-Responder"
    pred_str = "Responder" if prediction >= 0.5 else "Non-Responder"
    correct = (label == 1 and prediction >= 0.5) or (label == 0 and prediction < 0.5)
    status = "Correct" if correct else "Incorrect"
    
    fig.suptitle(
        f'Slide: {slide_id}\n'
        f'True: {label_str} | Predicted: {pred_str} (p={prediction:.3f}) | {status}',
        fontsize=14, y=1.02
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved attention visualization to {output_path}")


def load_model(model_path: Path):
    """Load trained CLAM model."""
    import torch
    import torch.nn as nn
    
    # Import training config and model
    from train_clam_production import CLAMProductionModel, TrainingConfig
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create config from saved config
    config_dict = checkpoint.get("config", {})
    config = TrainingConfig(**{k: v for k, v in config_dict.items() if hasattr(TrainingConfig, k)})
    
    model = CLAMProductionModel(config)
    model._setup_device()
    model._build_model()
    model._model.load_state_dict(checkpoint["model_state_dict"])
    model._model.to(model._device)
    model._model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    
    return model


def load_data(data_dir: Path, labels_file: str = "labels.csv"):
    """Load embeddings and labels."""
    labels_path = data_dir / labels_file
    labels_df = pd.read_csv(labels_path)
    
    id_col = "slide_id" if "slide_id" in labels_df.columns else "patient_id"
    
    embeddings_dir = data_dir / "embeddings"
    embeddings_dict = {}
    coords_dict = {}
    
    for _, row in labels_df.iterrows():
        slide_id = row[id_col]
        
        possible_names = [f"{slide_id}.npy", f"demo_{slide_id}.npy"]
        
        for name in possible_names:
            emb_path = embeddings_dir / name
            if emb_path.exists():
                embeddings_dict[slide_id] = np.load(emb_path)
                
                # Try to load coordinates
                coord_name = name.replace(".npy", "_coords.npy")
                coord_path = embeddings_dir / coord_name
                if coord_path.exists():
                    coords_dict[slide_id] = np.load(coord_path)
                break
    
    return embeddings_dict, coords_dict, labels_df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CLAM Model with Bootstrap CI and Visualizations"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/demo",
        help="Directory containing test data"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/evaluation",
        help="Output directory for results and plots"
    )
    parser.add_argument(
        "--n_bootstrap", type=int, default=1000,
        help="Number of bootstrap samples for CI"
    )
    parser.add_argument(
        "--n_attention_samples", type=int, default=5,
        help="Number of samples for attention visualization"
    )
    parser.add_argument(
        "--test_ids_file", type=str, default=None,
        help="Optional JSON file with test IDs (from training)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("CLAM Model Evaluation")
    logger.info("="*60)
    
    # Load model
    model_path = Path(args.model_path)
    model = load_model(model_path)
    
    # Load data
    data_dir = Path(args.data_dir)
    embeddings_dict, coords_dict, labels_df = load_data(data_dir)
    
    # Determine which samples to evaluate
    id_col = "slide_id" if "slide_id" in labels_df.columns else "patient_id"
    label_col = "label" if "label" in labels_df.columns else "treatment_response"
    
    if args.test_ids_file:
        with open(args.test_ids_file) as f:
            test_data = json.load(f)
            test_ids = test_data.get("test_ids", [])
    else:
        # Use all available data
        test_ids = [sid for sid in labels_df[id_col] if sid in embeddings_dict]
    
    logger.info(f"Evaluating on {len(test_ids)} samples")
    
    # Get predictions for all test samples
    y_true = []
    y_prob = []
    attentions = []
    
    for slide_id in test_ids:
        if slide_id not in embeddings_dict:
            continue
            
        embeddings = embeddings_dict[slide_id]
        prob, attn = model.predict(embeddings)
        
        row = labels_df[labels_df[id_col] == slide_id].iloc[0]
        label = row[label_col]
        if isinstance(label, str):
            label = 1 if label == "responder" else 0
        
        y_true.append(int(label))
        y_prob.append(prob)
        attentions.append((slide_id, attn))
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Compute metrics with bootstrap CI
    logger.info("\nComputing metrics with bootstrap confidence intervals...")
    metrics = compute_all_metrics_with_ci(
        y_true, y_prob,
        n_bootstrap=args.n_bootstrap
    )
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS (with 95% CI)")
    logger.info("="*60)
    
    for metric_name, metric_data in metrics.items():
        if metric_name == "confusion_matrix":
            cm = metric_data
            logger.info(f"\nConfusion Matrix:")
            logger.info(f"  TP: {cm['tp']}, TN: {cm['tn']}, FP: {cm['fp']}, FN: {cm['fn']}")
        else:
            logger.info(f"  {metric_name}: {metric_data['ci_95']}")
    
    # Calibration analysis
    logger.info("\nComputing calibration metrics...")
    calibration = compute_calibration(y_true, y_prob)
    logger.info(f"  Expected Calibration Error (ECE): {calibration['expected_calibration_error']:.4f}")
    logger.info(f"  Maximum Calibration Error (MCE): {calibration['maximum_calibration_error']:.4f}")
    
    metrics["calibration"] = calibration
    
    # Generate plots
    logger.info("\nGenerating visualizations...")
    
    # ROC curve
    plot_roc_curve(
        y_true, y_prob,
        output_dir / "roc_curve.png",
        n_bootstrap=100
    )
    
    # Calibration curve
    plot_calibration_curve(
        y_true, y_prob,
        output_dir / "calibration_curve.png"
    )
    
    # Attention visualizations
    attention_dir = output_dir / "attention_visualizations"
    attention_dir.mkdir(exist_ok=True)
    
    # Select samples for visualization (mix of correct/incorrect, responders/non-responders)
    n_viz = min(args.n_attention_samples, len(attentions))
    
    for i, (slide_id, attn) in enumerate(attentions[:n_viz]):
        coords = coords_dict.get(slide_id)
        
        # Get label and prediction
        row = labels_df[labels_df[id_col] == slide_id].iloc[0]
        label = row[label_col]
        if isinstance(label, str):
            label = 1 if label == "responder" else 0
        
        idx = list(test_ids).index(slide_id)
        prediction = y_prob[idx]
        
        plot_attention_visualization(
            embeddings_dict[slide_id],
            attn,
            coords,
            slide_id,
            label,
            prediction,
            attention_dir / f"attention_{slide_id}.png"
        )
    
    # Save results to JSON
    results = {
        "model_path": str(model_path),
        "n_samples": len(test_ids),
        "n_bootstrap": args.n_bootstrap,
        "metrics": {
            k: v if isinstance(v, dict) else v
            for k, v in metrics.items()
        },
        "predictions": {
            "slide_ids": test_ids,
            "y_true": y_true.tolist(),
            "y_prob": y_prob.tolist()
        }
    }
    
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {results_file}")
    logger.info(f"Plots saved to: {output_dir}")
    
    # Print summary table for paper
    logger.info("\n" + "="*60)
    logger.info("SUMMARY TABLE (for paper/submission)")
    logger.info("="*60)
    logger.info("\n| Metric | Value (95% CI) |")
    logger.info("|--------|----------------|")
    
    for metric_name in ["auc_roc", "sensitivity", "specificity", "ppv", "npv", "accuracy", "brier_score"]:
        if metric_name in metrics:
            ci_str = metrics[metric_name]["ci_95"]
            display_name = metric_name.replace("_", " ").title()
            logger.info(f"| {display_name} | {ci_str} |")
    
    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
