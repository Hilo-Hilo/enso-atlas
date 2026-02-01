#!/usr/bin/env python3
"""
Train CLAM Attention MIL model on TCGA Ovarian Cancer Treatment Response data.

Uses leave-one-out cross-validation given the small dataset (5 slides).
Saves best model to models/clam_ovarian.pt

Architecture matches src/enso_atlas/mil/clam.py CLAMClassifier exactly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import csv
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    input_dim: int = 384
    hidden_dim: int = 256
    attention_heads: int = 1
    dropout: float = 0.25
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 20


class GatedAttention(nn.Module):
    """Gated attention mechanism for CLAM - matches clam.py exactly."""

    def __init__(self, input_dim: int, hidden_dim: int, n_heads: int = 1, dropout: float = 0.25):
        super().__init__()
        self.n_heads = n_heads

        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(hidden_dim, n_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n_patches, input_dim)
        V = self.attention_V(x)  # (n_patches, hidden_dim)
        U = self.attention_U(x)  # (n_patches, hidden_dim)

        # Gated attention
        A = self.attention_weights(V * U)  # (n_patches, n_heads)
        A = torch.softmax(A, dim=0)  # Normalize over patches

        return A


class CLAMModel(nn.Module):
    """Full CLAM model with gated attention and instance clustering - matches clam.py exactly."""

    def __init__(self, input_dim: int, hidden_dim: int, n_heads: int, dropout: float, n_classes: int = 2):
        super().__init__()
        self.n_classes = n_classes

        # Feature transformation
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Gated attention
        self.attention = GatedAttention(hidden_dim, hidden_dim // 2, n_heads, dropout)

        # Instance-level classifier (for pseudo-labels)
        self.instance_classifier = nn.ModuleList([
            nn.Linear(hidden_dim, 2) for _ in range(n_classes)
        ])

        # Bag-level classifier
        self.bag_classifier = nn.Sequential(
            nn.Linear(hidden_dim * n_heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, return_attention: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Patch embeddings (n_patches, input_dim)
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (probability, attention_weights)
        """
        # Transform features
        h = self.feature_extractor(x)  # (n_patches, hidden_dim)

        # Compute attention
        A = self.attention(h)  # (n_patches, n_heads)

        # Aggregate with attention
        M = torch.mm(A.T, h)  # (n_heads, hidden_dim)
        M = M.view(-1)  # (n_heads * hidden_dim,)

        # Bag-level prediction
        prob = self.bag_classifier(M)

        if return_attention:
            # Average attention across heads
            attn_weights = A.mean(dim=1)  # (n_patches,)
            return prob, attn_weights

        return prob, None


def load_data(data_dir: Path) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Load embeddings and labels.

    Returns:
        Tuple of (embeddings_list, labels, slide_ids)
    """
    embeddings_dir = data_dir / "embeddings"
    labels_file = data_dir / "labels.csv"

    # Load labels
    labels_map = {}
    with open(labels_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            slide_file = row['slide_file'].replace('.svs', '.npy')
            treatment_response = row['treatment_response']
            labels_map[slide_file] = 1 if treatment_response == 'responder' else 0

    logger.info(f"Loaded labels for {len(labels_map)} slides")
    logger.info(f"Label distribution: {sum(labels_map.values())} responders, {len(labels_map) - sum(labels_map.values())} non-responders")

    # Load embeddings
    embeddings_list = []
    labels = []
    slide_ids = []

    for slide_file, label in labels_map.items():
        emb_path = embeddings_dir / slide_file
        if emb_path.exists():
            emb = np.load(emb_path).astype(np.float32)
            embeddings_list.append(emb)
            labels.append(label)
            slide_ids.append(slide_file.replace('.npy', ''))
            logger.info(f"  {slide_file}: {emb.shape[0]} patches, label={label} ({'responder' if label == 1 else 'non-responder'})")
        else:
            logger.warning(f"Embedding not found: {emb_path}")

    return embeddings_list, labels, slide_ids


def setup_device() -> torch.device:
    """Setup computation device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def train_model(
    model: nn.Module,
    train_embeddings: List[np.ndarray],
    train_labels: List[int],
    val_embeddings: List[np.ndarray],
    val_labels: List[int],
    config: TrainingConfig,
    device: torch.device
) -> Tuple[Dict[str, List[float]], float]:
    """
    Train the model.

    Returns:
        Tuple of (history, best_val_loss)
    """
    model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    criterion = nn.BCELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state_dict = None
    patience_counter = 0

    for epoch in range(config.epochs):
        # Training
        model.train()
        train_losses = []

        indices = np.random.permutation(len(train_embeddings))

        for idx in indices:
            x = torch.from_numpy(train_embeddings[idx]).float().to(device)
            y = torch.tensor([train_labels[idx]], dtype=torch.float32).to(device)

            optimizer.zero_grad()
            pred, _ = model(x)
            loss = criterion(pred.squeeze(), y.squeeze())
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for emb, label in zip(val_embeddings, val_labels):
                x = torch.from_numpy(emb).float().to(device)
                y = torch.tensor([label], dtype=torch.float32).to(device)
                pred, _ = model(x)
                loss = criterion(pred.squeeze(), y.squeeze())
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        history["val_loss"].append(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{config.epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return history, best_val_loss


def leave_one_out_cv(
    embeddings_list: List[np.ndarray],
    labels: List[int],
    slide_ids: List[str],
    config: TrainingConfig,
    device: torch.device
) -> Tuple[List[float], List[float], List[Dict]]:
    """
    Perform leave-one-out cross-validation.

    Returns:
        Tuple of (predictions, true_labels, attention_weights_list)
    """
    n_samples = len(embeddings_list)
    predictions = []
    true_labels = []
    attention_results = []

    logger.info(f"\n{'='*60}")
    logger.info("LEAVE-ONE-OUT CROSS-VALIDATION")
    logger.info(f"{'='*60}")

    for i in range(n_samples):
        logger.info(f"\nFold {i+1}/{n_samples}: Testing on {slide_ids[i]}")

        # Split data
        train_emb = [embeddings_list[j] for j in range(n_samples) if j != i]
        train_lab = [labels[j] for j in range(n_samples) if j != i]
        test_emb = embeddings_list[i]
        test_lab = labels[i]

        # Build model
        model = CLAMModel(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            n_heads=config.attention_heads,
            dropout=config.dropout
        )

        # Train (use training data as validation for early stopping)
        history, best_loss = train_model(
            model, train_emb, train_lab,
            train_emb, train_lab,  # Use train as val for LOO
            config, device
        )

        # Predict
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(test_emb).float().to(device)
            pred, attn = model(x)
            pred_prob = pred.item()
            attn_weights = attn.cpu().numpy()

        predictions.append(pred_prob)
        true_labels.append(test_lab)
        attention_results.append({
            'slide_id': slide_ids[i],
            'prediction': pred_prob,
            'true_label': test_lab,
            'attention_weights': attn_weights
        })

        pred_class = 1 if pred_prob > 0.5 else 0
        correct = "CORRECT" if pred_class == test_lab else "WRONG"
        logger.info(f"  Prediction: {pred_prob:.4f} ({'responder' if pred_class == 1 else 'non-responder'})")
        logger.info(f"  True label: {test_lab} ({'responder' if test_lab == 1 else 'non-responder'})")
        logger.info(f"  Result: {correct}")

    return predictions, true_labels, attention_results


def train_final_model(
    embeddings_list: List[np.ndarray],
    labels: List[int],
    config: TrainingConfig,
    device: torch.device
) -> CLAMModel:
    """Train final model on all data."""
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING FINAL MODEL ON ALL DATA")
    logger.info(f"{'='*60}")

    model = CLAMModel(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        n_heads=config.attention_heads,
        dropout=config.dropout
    )

    # Train on all data
    history, _ = train_model(
        model, embeddings_list, labels,
        embeddings_list, labels,
        config, device
    )

    return model


def evaluate_predictions(predictions: List[float], true_labels: List[int], attention_results: List[Dict]):
    """Evaluate and print results."""
    logger.info(f"\n{'='*60}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'='*60}")

    # Calculate accuracy
    pred_classes = [1 if p > 0.5 else 0 for p in predictions]
    accuracy = sum(p == t for p, t in zip(pred_classes, true_labels)) / len(true_labels)

    logger.info(f"\nAccuracy: {accuracy:.2%} ({sum(p == t for p, t in zip(pred_classes, true_labels))}/{len(true_labels)})")

    # Per-class breakdown
    responders = [(p, t) for p, t in zip(predictions, true_labels) if t == 1]
    non_responders = [(p, t) for p, t in zip(predictions, true_labels) if t == 0]

    logger.info(f"\nResponder slides (should predict >0.5):")
    for r in attention_results:
        if r['true_label'] == 1:
            logger.info(f"  {r['slide_id']}: {r['prediction']:.4f}")

    logger.info(f"\nNon-responder slides (should predict <0.5):")
    for r in attention_results:
        if r['true_label'] == 0:
            logger.info(f"  {r['slide_id']}: {r['prediction']:.4f}")

    # Attention weight analysis
    logger.info(f"\nAttention Weight Statistics:")
    for r in attention_results:
        attn = r['attention_weights']
        logger.info(f"  {r['slide_id']}: min={attn.min():.4f}, max={attn.max():.4f}, "
                   f"std={attn.std():.4f}, top_patch_attn={attn.max():.4f}")


def main():
    """Main training function."""
    # Paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"
    models_dir = project_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # Config
    config = TrainingConfig(
        input_dim=384,
        hidden_dim=256,
        attention_heads=1,
        dropout=0.25,
        learning_rate=0.001,
        weight_decay=1e-4,
        epochs=100,
        patience=20
    )

    logger.info("CLAM Training for Ovarian Cancer Treatment Response")
    logger.info(f"Config: {config}")

    # Setup
    device = setup_device()

    # Load data
    embeddings_list, labels, slide_ids = load_data(data_dir)

    if len(embeddings_list) == 0:
        logger.error("No data found!")
        return

    # Cross-validation
    predictions, true_labels, attention_results = leave_one_out_cv(
        embeddings_list, labels, slide_ids, config, device
    )

    # Evaluate
    evaluate_predictions(predictions, true_labels, attention_results)

    # Train final model on all data
    final_model = train_final_model(embeddings_list, labels, config, device)

    # Save model
    output_path = models_dir / "clam_ovarian.pt"
    torch.save(final_model.state_dict(), output_path)
    logger.info(f"\nSaved trained model to {output_path}")

    # Verify model loads correctly with the existing CLAMClassifier
    logger.info("\nVerifying model compatibility with CLAMClassifier...")
    from enso_atlas.config import MILConfig
    from enso_atlas.mil.clam import CLAMClassifier

    mil_config = MILConfig(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        attention_heads=config.attention_heads,
        dropout=config.dropout
    )
    classifier = CLAMClassifier(mil_config)
    classifier.load(output_path)
    logger.info("Model verified: compatible with CLAMClassifier")

    # Final predictions with trained model
    logger.info(f"\n{'='*60}")
    logger.info("FINAL MODEL PREDICTIONS (via CLAMClassifier)")
    logger.info(f"{'='*60}")

    for emb, label, slide_id in zip(embeddings_list, labels, slide_ids):
        prob, attn = classifier.predict(emb)
        pred_class = "responder" if prob > 0.5 else "non-responder"
        true_class = "responder" if label == 1 else "non-responder"
        logger.info(f"  {slide_id}: pred={prob:.4f} ({pred_class}), true={true_class}")


if __name__ == "__main__":
    main()
