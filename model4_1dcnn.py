import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Conv1D, GlobalMaxPooling1D,
                                      Dense, Dropout, Concatenate, SpatialDropout1D)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import re, time, json

tf.random.set_seed(42)
np.random.seed(42)


VOCAB_SIZE   = 20_000
MAX_LEN      = 300
EMBED_DIM    = 128
NUM_FILTERS  = 128
KERNEL_SIZES = [3, 4, 5]
BATCH_SIZE   = 64
MAX_EPOCHS   = 20


print("Loading dataset...")
df = pd.read_csv("IMDB Dataset.csv")
df["label"] = df["sentiment"].map({"negative": 0, "positive": 1})

def clean(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text.lower().strip()

df["clean"] = df["review"].apply(clean)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df["clean"].values, df["label"].values,
    test_size=0.2, random_state=42, stratify=df["label"].values)


print("Tokenizing...")
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_raw)
X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_raw), maxlen=MAX_LEN, truncating="post")
X_test  = pad_sequences(tokenizer.texts_to_sequences(X_test_raw),  maxlen=MAX_LEN, truncating="post")


inp = Input(shape=(MAX_LEN,))
emb = Embedding(VOCAB_SIZE, EMBED_DIM)(inp)
emb = SpatialDropout1D(0.3)(emb)

branches = []
for k in KERNEL_SIZES:
    x = Conv1D(NUM_FILTERS, k, activation="relu", padding="valid")(emb)
    x = GlobalMaxPooling1D()(x)
    branches.append(x)

concat = Concatenate()(branches)
x = Dense(128, activation="relu")(concat)
x = Dropout(0.5)(x)
out = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
]

print("\nTraining 1D CNN (TextCNN)...")
start = time.time()
history = model.fit(
    X_train, y_train,
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1,
)
train_time = time.time() - start

loss, acc = model.evaluate(X_test, y_test, verbose=0)
y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()

report = classification_report(y_test, y_pred, target_names=["Negative","Positive"])
cm     = confusion_matrix(y_test, y_pred)

print(f"\n{'='*50}")
print(f"Test Loss:     {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")
print(f"\nClassification Report:\n{report}")
print(f"Confusion Matrix:\n{cm}")

model.save("model4_1dcnn.keras")

results = {
    "model": "1D CNN (TextCNN)",
    "test_accuracy": float(acc),
    "test_loss": float(loss),
    "epochs_trained": len(history.history["loss"]),
    "training_time_seconds": round(train_time, 1),
    "vocab_size": VOCAB_SIZE,
    "max_sequence_length": MAX_LEN,
    "embedding_dim": EMBED_DIM,
    "num_filters": NUM_FILTERS,
    "kernel_sizes": KERNEL_SIZES,
    "baseline_accuracy": 0.775,
    "improvement": round(float(acc) - 0.775, 4),
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

with open("results.txt", "w") as f:
    f.write("MODEL 4: 1D CNN (TextCNN) RESULTS\n")
    f.write("="*50 + "\n\n")
    f.write(f"Test Accuracy:  {acc:.4f}\n")
    f.write(f"Test Loss:      {loss:.4f}\n")
    f.write(f"Epochs Trained: {len(history.history['loss'])}\n")
    f.write(f"Training Time:  {train_time:.1f}s\n")
    f.write(f"Baseline (hw1): 0.7750\n")
    f.write(f"Improvement:    {acc - 0.775:+.4f}\n\n")
    f.write("Epoch History:\n")
    for i, (tl, ta, vl, va) in enumerate(zip(
        history.history["loss"], history.history["accuracy"],
        history.history["val_loss"], history.history["val_accuracy"]
    ), 1):
        f.write(f"  Epoch {i:2d}: loss={tl:.4f} acc={ta:.4f} | val_loss={vl:.4f} val_acc={va:.4f}\n")
    f.write(f"\nClassification Report:\n{report}")
    f.write(f"\nConfusion Matrix:\n{cm}\n")

print("\nSaved: model4_1dcnn.keras, results.txt, results.json")
