# Model 4: 1D CNN (TextCNN)

## Overview
Inspired by Kim (2014) *"Convolutional Neural Networks for Sentence Classification"*, this model applies multiple 1D convolutional filters in parallel over the embedded token sequence. Each filter acts like an n-gram detector, and combining multiple kernel sizes captures local patterns at different granularities. CNNs train significantly faster than RNNs and are highly parallelizable.

## Model Architecture

```
Input (token sequence, max_len=300)
  └─► Embedding(vocab=20000, dim=128)
        └─► SpatialDropout1D(0.3)
              ├─► Conv1D(128 filters, kernel=3, relu) → GlobalMaxPooling1D ─┐
              ├─► Conv1D(128 filters, kernel=4, relu) → GlobalMaxPooling1D ─┤
              └─► Conv1D(128 filters, kernel=5, relu) → GlobalMaxPooling1D ─┘
                          Concatenate (384 units)
                              └─► Dense(128, relu)
                                    └─► Dropout(0.5)
                                          └─► Dense(1, sigmoid)
```

**Total parameters:** ~3.8M

## Techniques Applied

### 1. Multi-Kernel Parallel Convolutions
Three parallel Conv1D branches with kernel sizes **3, 4, 5** are applied simultaneously on the same embedding. This lets the model detect:
- Trigrams (kernel=3): short phrase patterns
- 4-grams (kernel=4): medium-length idioms
- 5-grams (kernel=5): longer sentiment phrases

### 2. GlobalMaxPooling1D
Each conv branch outputs the single most salient activation across the entire sequence — the strongest n-gram match — before merging. This makes the representation sequence-length invariant.

### 3. Feature Concatenation
The three pooled outputs (3 × 128 = 384 units) are concatenated to form a rich multi-scale representation.

### 4. SpatialDropout1D
Applied on the embedding output, dropping full word vector channels to regularize the embedding more effectively than element-wise Dropout.

### 5. High Dropout (0.5) Before Output
Aggressive dropout in the dense head forces the model to learn redundant representations and prevents co-adaptation.

### 6. EarlyStopping + ReduceLROnPlateau
Standard callbacks for stable training.

## Why CNNs for Text?

| Property | CNN | RNN/LSTM |
|----------|-----|----------|
| Training speed | Very fast | Slow (sequential) |
| Parallelism | High | Low |
| Long-range dependencies | Limited | Captures well |
| Local patterns (n-grams) | Excellent | Moderate |
| Interpretability | Higher | Lower |

For sentiment classification where n-gram patterns ("not good", "absolutely terrible", "highly recommended") are diagnostic, CNNs are extremely effective.

## Reference
Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. *EMNLP 2014*.

## How to Run

```bash
# Place IMDB Dataset.csv in the same directory
pip install tensorflow scikit-learn pandas numpy
python model4_1dcnn.py
```

## Files

| File | Description |
|------|-------------|
| `model4_1dcnn.py` | Training script |
| `results.txt` | Full training log + metrics |
| `results.json` | Machine-readable results summary |
| `model4_1dcnn.keras` | Saved model weights |
