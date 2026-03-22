# Model 3: Bidirectional GRU

## Overview
The GRU (Gated Recurrent Unit) is a streamlined alternative to the LSTM. It merges the forget and input gates into a single "update gate" and has no separate cell state, making it faster to train while achieving comparable accuracy on many NLP tasks. This model adds `SpatialDropout1D` on the embedding for stronger embedding regularization.

## Model Architecture

```
Input (token sequence, max_len=300)
  └─► Embedding(vocab=20000, dim=100)
        └─► SpatialDropout1D(0.3)
              └─► Bidirectional GRU(64, return_sequences=True, dropout=0.2)
                    └─► Bidirectional GRU(32)
                          └─► Dense(64, relu)
                                └─► Dropout(0.4)
                                      └─► Dense(1, sigmoid)
```

**Total parameters:** ~3.0M (lighter than the LSTM model)

## Techniques Applied

### 1. GRU vs LSTM
- GRU has fewer parameters per unit (2 gates vs 3 in LSTM), making it faster to train
- Competitive on sentiment tasks where very long-term dependencies are less critical
- Smaller embedding dim (100 vs 128) compensates for the lighter recurrent unit

### 2. SpatialDropout1D
Applied directly on the Embedding output. Drops entire feature maps (word vectors) rather than individual elements, making it more effective at regularizing the embedding layer than standard Dropout.

### 3. Bidirectional Wrapping
Both GRU layers are wrapped in `Bidirectional`, doubling their hidden state capacity and allowing the model to consider both past and future context when predicting sentiment.

### 4. Stacked GRU
Two layers of Bidirectional GRU allow higher-level sentiment abstractions to emerge from lower-level token patterns.

### 5. EarlyStopping + ReduceLROnPlateau
- Restores best weights automatically
- Adaptive learning rate decay on plateau

## GRU vs LSTM Comparison

| Property | LSTM | GRU |
|----------|------|-----|
| Gates | 3 (forget, input, output) | 2 (update, reset) |
| Cell state | Separate | Merged into hidden state |
| Parameters | More | Fewer |
| Training speed | Slower | Faster |
| Typical accuracy | Slightly higher | Comparable |

## How to Run

```bash
# Place IMDB Dataset.csv in the same directory
pip install tensorflow scikit-learn pandas numpy
python model3_gru.py
```

## Files

| File | Description |
|------|-------------|
| `model3_gru.py` | Training script |
| `results.txt` | Full training log + metrics |
| `results.json` | Machine-readable results summary |
| `model3_gru.keras` | Saved model weights |
