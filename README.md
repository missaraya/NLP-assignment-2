# Model 5: Word2Vec Embeddings + Bidirectional LSTM

## Overview
This model demonstrates the use of **pretrained word embeddings**. A Word2Vec model is trained on the full IMDB corpus using the Gensim library, and the resulting 100-dimensional word vectors are used to initialize the Keras Embedding layer. Training is performed in two phases: first with frozen embeddings (preserving the semantic geometry learned by Word2Vec), then with fine-tuning at a lower learning rate.

## Model Architecture

```
Input (token sequence, max_len=300)
  └─► Embedding(vocab=20000, dim=100)  ← initialized from Word2Vec matrix
        └─► SpatialDropout1D(0.3)
              └─► Bidirectional LSTM(64, return_sequences=True, dropout=0.3)
                    └─► Bidirectional LSTM(32, dropout=0.2)
                          └─► Dense(64, relu)
                                └─► Dropout(0.4)
                                      └─► Dense(1, sigmoid)
```

**Total parameters:** ~2.7M

## Two-Phase Training Strategy

### Phase 1 — Frozen Embeddings (lr=1e-3)
The embedding layer is frozen (`trainable=False`). This forces the LSTM layers to learn from the rich Word2Vec geometry without destroying it with noisy gradients in the early epochs. `EarlyStopping` with `patience=3` terminates this phase when validation accuracy plateaus.

### Phase 2 — Fine-Tuning (lr=1e-4)
The embedding layer is unfrozen and the model is compiled with a 10× lower learning rate. This allows small, targeted adjustments to the word vectors for the specific sentiment task, without catastrophically forgetting the general-purpose representations from Phase 1.

## Techniques Applied

### 1. Word2Vec Training with Gensim
```
Word2Vec(vector_size=100, window=5, min_count=2, epochs=10)
```
- Trained on all 50,000 IMDB reviews (train + test split together for richer vocabulary)
- `window=5`: considers ±5 word context for each target word
- `min_count=2`: removes hapax legomena that appear only once

### 2. Embedding Matrix Construction
- The Keras `Tokenizer` word index is aligned with the Gensim W2V vocabulary
- For each word in the tokenizer's top-20k: if it exists in W2V, its vector is inserted; otherwise the row stays zero-initialized
- Typical hit rate: >85%

### 3. Transfer Learning Paradigm
This two-phase approach mirrors BERT fine-tuning: a strong prior (Word2Vec) is preserved initially, then carefully adapted.

### 4. SpatialDropout1D
Regularizes the embedding output by dropping entire feature dimensions.

### 5. Stacked Bidirectional LSTM
Captures both short-term and long-range dependencies in both directions.

## Word2Vec vs Random Initialization

| Property | Word2Vec Init | Random Init |
|----------|--------------|-------------|
| Convergence speed | Faster | Slower |
| Final accuracy (small data) | Higher | Lower |
| Semantic geometry | Pre-structured | Learned from scratch |
| Fine-tuning benefit | High | N/A |

## How to Run

```bash
# Place IMDB Dataset.csv in the same directory
pip install tensorflow gensim scikit-learn pandas numpy
python model5_word2vec_lstm.py
```

## Files

| File | Description |
|------|-------------|
| `model5_word2vec_lstm.py` | Training script |
| `results.txt` | Full training log + metrics |
| `results.json` | Machine-readable results summary |
| `model5_word2vec_lstm.keras` | Saved Keras model |
| `word2vec_imdb.model` | Trained Gensim Word2Vec model |
