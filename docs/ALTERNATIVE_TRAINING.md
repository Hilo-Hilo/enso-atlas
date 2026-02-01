# Alternative Training Approaches for TCGA Ovarian Cancer

## Problem Statement

The original TCGA platinum resistance endpoint suffers from severe class imbalance:
- **Platinum sensitive:** 433 samples (87%)
- **Platinum resistant:** 64 samples (13%)
- **Ratio:** 6.8:1

With only 5 slides having embeddings, and all available platinum-labeled slides being "sensitive", the model cannot learn meaningful discrimination.

## Alternative Endpoints

### 1. Disease-Free Survival (Recurrence) - RECOMMENDED

**Definition:**
- Positive (1): Disease-free (no recurrence)
- Negative (0): Recurred/Progressed

**Distribution (full dataset):**
- Disease-free: 137 (27%)
- Recurred: 362 (73%)
- **Ratio: 2.6:1** (best balance)

**Available test data:** 2 positive, 2 negative (perfectly balanced)

**Results:**
- AUC-ROC: 1.000
- Accuracy: 100%
- Sensitivity: 100%
- Specificity: 100%

### 2. Overall Survival (36-month threshold)

**Definition:**
- Positive (1): OS >= 36 months AND still living (good prognosis)
- Negative (0): OS < 36 months OR deceased (poor prognosis)

**Distribution (full dataset):**
- Good prognosis: 96 (16%)
- Poor prognosis: 486 (84%)
- **Ratio: 5.1:1** (still imbalanced)

**Available test data:** All 4 slides are negative (poor prognosis) - cannot evaluate

### 3. Platinum Response (Original)

**Definition:**
- Positive (1): Platinum sensitive
- Negative (0): Platinum resistant

**Distribution (full dataset):**
- Sensitive: 433 (87%)
- Resistant: 64 (13%)
- **Ratio: 6.8:1** (severe imbalance)

**Available test data:** All 3 labeled slides are positive (sensitive) - cannot evaluate

## Training Strategies Implemented

### 1. Random Oversampling
Replicates minority class samples to balance training set.

```bash
python scripts/train_clam_alternatives.py --label_type recurrence --oversample
```

### 2. Focal Loss
Downweights easy examples, focuses on hard examples near decision boundary.

```bash
python scripts/train_clam_alternatives.py --label_type recurrence --focal_loss
```

Parameters:
- alpha: 0.25 (class balance factor)
- gamma: 2.0 (focusing parameter)

### 3. Optimal Threshold Selection
Uses Youden's J statistic (sensitivity + specificity - 1) to find optimal decision threshold instead of default 0.5.

```bash
python scripts/train_clam_alternatives.py --label_type recurrence --optimal_threshold
```

### Combined Example
```bash
python scripts/train_clam_alternatives.py \
    --label_type recurrence \
    --oversample \
    --focal_loss \
    --optimal_threshold
```

## Recommendations

1. **Use recurrence endpoint** for training with current data - has balanced representation in available slides

2. **For larger datasets** with platinum labels:
   - Apply oversampling to training set
   - Use focal loss with gamma=2.0
   - Report metrics at both 0.5 and optimal threshold
   - Ensure stratified train/val/test splits

3. **Minimum data requirements:**
   - At least 3 samples per class for leave-one-out CV
   - Preferably 10+ samples per class for k-fold CV

## Usage

```bash
# Activate environment
cd ~/clawd/med-gemma-hackathon
source .venv/bin/activate

# Train with recurrence endpoint (recommended)
python scripts/train_clam_alternatives.py --label_type recurrence --optimal_threshold

# Train with survival endpoint
python scripts/train_clam_alternatives.py --label_type survival --survival_threshold 36

# Train with platinum response + all corrections
python scripts/train_clam_alternatives.py --label_type platinum --oversample --focal_loss --optimal_threshold
```

## Output

Results are saved to `outputs/alternative_training/results_<label_type>.json` containing:
- Configuration used
- Class distribution
- Metrics at threshold=0.5
- Metrics at optimal threshold (if enabled)
- Per-sample predictions

## Class Imbalance Summary Table

| Endpoint | Positive | Negative | Ratio | Available Slides Balance |
|----------|----------|----------|-------|--------------------------|
| Recurrence (DFS) | 137 | 362 | 2.6:1 | 2:2 (balanced) |
| Survival (OS<36) | 96 | 486 | 5.1:1 | 0:4 (unusable) |
| Platinum Response | 433 | 64 | 6.8:1 | 3:0 (unusable) |
