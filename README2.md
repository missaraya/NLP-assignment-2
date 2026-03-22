# Model 2: Bidirectional LSTM

## Overview
Unlike the MLP models that rely on hand-crafted features, this model learns directly from raw text using a sequence model. A Bidirectional LSTM reads each review forwards and backwards, capturing long-range dependencies in movie reviews that rule-based sentiment scorers miss.

## Model Architecture

```
Input (token sequence, max_len=300)
  └─► Embedding(vocab=20000, dim=128)
        └─► Bidirectional LSTM(64, return_sequences=True, dropout=0.3)
              └─► Bidirectional LSTM(32)
                    └─► Dense(64, relu)
                          └─► Dropout(0.4)
                                └─► Dense(1, sigmoid)
```

**Total parameters:** ~4.2M (dominated by the Embedding layer)

## Techniques Applied

### 1. Text Preprocessing
- HTML tag stripping (reviews contain `<br />` etc.)
- Lowercasing + non-alphabetic character removal
- Keras `Tokenizer` with `VOCAB_SIZE=20,000` and OOV token

### 2. Sequence Padding
All sequences are padded/truncated to `MAX_LEN=300` tokens, covering ~95% of reviews without excessive padding.

### 3. Trainable Embedding Layer
A randomly initialized Embedding layer (dim=128) is trained end-to-end, learning task-specific word representations.

### 4. Bidirectional LSTM
- Two stacked Bidirectional LSTM layers
- `return_sequences=True` on the first layer allows the second layer to process the full sequence
- Internal dropout (`dropout`, `recurrent_dropout`) on both layers regularizes the recurrent weights

### 5. Stacked LSTM Architecture
Two LSTM layers allow higher-level abstract representations of sentiment patterns to emerge.

### 6. EarlyStopping + ReduceLROnPlateau
- Stops training when validation accuracy stops improving (patience=3)
- Halves learning rate on plateau (factor=0.5, patience=2)

## How to Run

```bash
# Place IMDB Dataset.csv in the same directory
pip install tensorflow scikit-learn pandas numpy
python model2_lstm.py
```

## Files

| File | Description |
|------|-------------|
| `model2_lstm.py` | Training script |
| `results.txt` | Full training log + metrics |
| `results.json` | Machine-readable results summary |
| `model2_lstm.keras` | Saved model weights |
