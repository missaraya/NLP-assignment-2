# Model 1: Optimized MLP

## Overview
An improved version of the Assignment 1 baseline MLP, applying several optimization techniques discussed in class to push accuracy beyond the 77.5% ceiling of the original two-feature model.

## Model Architecture

```
Input (10 features)
  └─► Dense(64, relu)
        └─► BatchNormalization
              └─► Dropout(0.3)
                    └─► Dense(32, relu)
                          └─► BatchNormalization
                                └─► Dropout(0.3)
                                      └─► Dense(16, relu)
                                            └─► Dropout(0.2)
                                                  └─► Dense(1, sigmoid)
```

**Total parameters:** ~3,500

## Techniques Applied

### 1. Extended Feature Engineering
The baseline used only 2 features (VADER compound + TextBlob polarity). This model uses **10 features**:

| Feature | Description |
|---------|-------------|
| `vader_compound` | VADER overall sentiment score |
| `vader_pos` | VADER positive component |
| `vader_neg` | VADER negative component |
| `vader_neu` | VADER neutral component |
| `blob_polarity` | TextBlob polarity |
| `blob_subjectivity` | TextBlob subjectivity |
| `word_count` | Number of words in review |
| `char_count` | Character length of review |
| `excl_count` | Number of exclamation marks |
| `ques_count` | Number of question marks |

### 2. Batch Normalization
Applied after each Dense layer to stabilize activations and accelerate convergence.

### 3. Dropout Regularization
Rates of 0.3 and 0.2 prevent overfitting on the relatively low-dimensional feature space.

### 4. Learning Rate Scheduling
`ReduceLROnPlateau` halves the learning rate when validation loss plateaus, allowing fine-grained convergence at the end of training.

### 5. Early Stopping
Monitors `val_loss` with `patience=5`; restores best weights automatically to avoid overfitting.

### 6. Wider & Deeper Architecture
Increased from 16→8→1 (baseline) to 64→32→16→1 with normalized activations.

## How to Run

```bash
# Place IMDB Dataset.csv in the same directory
pip install tensorflow textblob vaderSentiment scikit-learn pandas numpy
python model1_optimized_mlp.py
```

## Files

| File | Description |
|------|-------------|
| `model1_optimized_mlp.py` | Training script |
| `results.txt` | Full training log + metrics |
| `results.json` | Machine-readable results summary |
| `model1_optimized_mlp.keras` | Saved model weights |
