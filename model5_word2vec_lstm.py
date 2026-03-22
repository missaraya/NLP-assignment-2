import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, Bidirectional, LSTM,
                                      Dense, Dropout, SpatialDropout1D, Input)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from gensim.models import Word2Vec
import re, time, json

tf.random.set_seed(42)
np.random.seed(42)

VOCAB_SIZE  = 20_000
MAX_LEN     = 300
EMBED_DIM   = 100        
W2V_WINDOW  = 5
W2V_MINCOUNT = 2
BATCH_SIZE  = 64
MAX_EPOCHS  = 20

print("Loading dataset...")
df = pd.read_csv("IMDB Dataset.csv")
df["label"] = df["sentiment"].map({"negative": 0, "positive": 1})

def clean(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text.lower().strip()

df["clean"] = df["review"].apply(clean)
df["tokens"] = df["clean"].apply(str.split)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df["clean"].values, df["label"].values,
    test_size=0.2, random_state=42, stratify=df["label"].values)


print("Training Word2Vec on the entire corpus...")
all_sentences = df["tokens"].tolist()
w2v_model = Word2Vec(
    sentences=all_sentences,
    vector_size=EMBED_DIM,
    window=W2V_WINDOW,
    min_count=W2V_MINCOUNT,
    workers=4,
    epochs=10,
    seed=42,
)
print(f"  Vocabulary size: {len(w2v_model.wv):,}")


print("Tokenizing for Keras...")
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_raw)
X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_raw), maxlen=MAX_LEN, truncating="post")
X_test  = pad_sequences(tokenizer.texts_to_sequences(X_test_raw),  maxlen=MAX_LEN, truncating="post")

word_index = tokenizer.word_index
embedding_matrix = np.zeros((VOCAB_SIZE, EMBED_DIM))
hits, misses = 0, 0
for word, idx in word_index.items():
    if idx >= VOCAB_SIZE:
        continue
    if word in w2v_model.wv:
        embedding_matrix[idx] = w2v_model.wv[word]
        hits += 1
    else:
        misses += 1

print(f"  Embedding hits: {hits:,}  misses: {misses:,}")


model = Sequential([
    Input(shape=(MAX_LEN,)),
    Embedding(
        VOCAB_SIZE, EMBED_DIM,
        embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
        trainable=False,          # frozen initially
    ),
    SpatialDropout1D(0.3),
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),
    Bidirectional(LSTM(32, dropout=0.2)),
    Dense(64, activation="relu"),
    Dropout(0.4),
    Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()

print("\nPhase 1: Training with frozen Word2Vec embeddings...")
callbacks_p1 = [
    EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
]
start = time.time()
history1 = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=callbacks_p1,
    verbose=1,
)


print("\nPhase 2: Fine-tuning (embedding layer unfrozen)...")
model.layers[1].trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),   # lower LR for fine-tuning
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
callbacks_p2 = [
    EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
]
history2 = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=callbacks_p2,
    verbose=1,
)
train_time = time.time() - start


all_hist_keys = ["loss", "accuracy", "val_loss", "val_accuracy"]
combined_history = {k: history1.history[k] + history2.history[k] for k in all_hist_keys}


loss, acc = model.evaluate(X_test, y_test, verbose=0)
y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()

report = classification_report(y_test, y_pred, target_names=["Negative","Positive"])
cm     = confusion_matrix(y_test, y_pred)

print(f"\n{'='*50}")
print(f"Test Loss:     {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")
print(f"\nClassification Report:\n{report}")
print(f"Confusion Matrix:\n{cm}")


model.save("model5_word2vec_lstm.keras")
w2v_model.save("word2vec_imdb.model")

results = {
    "model": "Word2Vec + BiLSTM",
    "test_accuracy": float(acc),
    "test_loss": float(loss),
    "total_epochs": len(combined_history["loss"]),
    "training_time_seconds": round(train_time, 1),
    "word2vec_vector_size": EMBED_DIM,
    "word2vec_window": W2V_WINDOW,
    "embedding_hits": hits,
    "embedding_misses": misses,
    "baseline_accuracy": 0.775,
    "improvement": round(float(acc) - 0.775, 4),
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

with open("results.txt", "w") as f:
    f.write("MODEL 5: WORD2VEC + BILSTM RESULTS\n")
    f.write("="*50 + "\n\n")
    f.write(f"Test Accuracy:  {acc:.4f}\n")
    f.write(f"Test Loss:      {loss:.4f}\n")
    f.write(f"Total Epochs:   {len(combined_history['loss'])}\n")
    f.write(f"Training Time:  {train_time:.1f}s\n")
    f.write(f"Baseline (hw1): 0.7750\n")
    f.write(f"Improvement:    {acc - 0.775:+.4f}\n\n")
    f.write(f"Word2Vec Config:\n")
    f.write(f"  vector_size={EMBED_DIM}, window={W2V_WINDOW}, min_count={W2V_MINCOUNT}\n")
    f.write(f"  Embedding hits: {hits:,}  misses: {misses:,}\n\n")
    f.write("Epoch History:\n")
    for i, (tl, ta, vl, va) in enumerate(zip(
        combined_history["loss"], combined_history["accuracy"],
        combined_history["val_loss"], combined_history["val_accuracy"]
    ), 1):
        f.write(f"  Epoch {i:2d}: loss={tl:.4f} acc={ta:.4f} | val_loss={vl:.4f} val_acc={va:.4f}\n")
    f.write(f"\nClassification Report:\n{report}")
    f.write(f"\nConfusion Matrix:\n{cm}\n")

print("\nSaved: model5_word2vec_lstm.keras, results.txt, results.json, word2vec_imdb.model")
