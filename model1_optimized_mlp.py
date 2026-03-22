
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time, json

np.random.seed(42)


print("Loading dataset...")
df = pd.read_csv("IMDB Dataset.csv")
print(f"  Rows: {len(df)}")


def get_features(text):
    scores = analyzer.polarity_scores(text)
    vader_compound = scores["compound"]
    vader_pos      = scores["pos"]
    vader_neg      = scores["neg"]
    vader_neu      = scores["neu"]
    blob_polarity  = TextBlob(text).sentiment.polarity
    blob_subj      = TextBlob(text).sentiment.subjectivity
    word_count     = len(text.split())
    char_count     = len(text)
    excl_count     = text.count("!")
    ques_count     = text.count("?")
    return pd.Series([vader_compound, vader_pos, vader_neg, vader_neu,
                      blob_polarity, blob_subj,
                      word_count, char_count, excl_count, ques_count])

print("Extracting features (this may take a minute)...")
feature_cols = ["vader_compound","vader_pos","vader_neg","vader_neu",
                "blob_polarity","blob_subjectivity",
                "word_count","char_count","excl_count","ques_count"]
df[feature_cols] = df["review"].apply(get_features)
df["label"] = df["sentiment"].map({"negative": 0, "positive": 1})

X = df[feature_cols].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Model ─────────────────────────────────────────────────────────────────────
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
]

# ── Training ──────────────────────────────────────────────────────────────────
print("\nTraining Optimized MLP...")
start = time.time()
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1,
)
train_time = time.time() - start
loss, acc = model.evaluate(X_test, y_test, verbose=0)
y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()

report = classification_report(y_test, y_pred, target_names=["Negative","Positive"])
cm = confusion_matrix(y_test, y_pred)

print(f"\n{'='*50}")
print(f"Test Loss:     {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")
print(f"\nClassification Report:\n{report}")
print(f"Confusion Matrix:\n{cm}")
print(f"Training time: {train_time:.1f}s")

model.save("model1_optimized_mlp.keras")
results = {
    "model": "Optimized MLP",
    "test_accuracy": float(acc),
    "test_loss": float(loss),
    "epochs_trained": len(history.history["loss"]),
    "training_time_seconds": round(train_time, 1),
    "baseline_accuracy": 0.775,
    "improvement": round(float(acc) - 0.775, 4),
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

with open("results.txt", "w") as f:
    f.write("MODEL 1: OPTIMIZED MLP RESULTS\n")
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

print("\nSaved: model1_optimized_mlp.keras, results.txt, results.json")
