"""Train and save the Stage 1 ML classifier (TF-IDF + LinearSVC).

Usage:
    python -m tools.train_classifier

Output:
    data/classifiers/stage1_model.pkl  — trained pipeline (TfidfVectorizer + LinearSVC)
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


FUNCTION_NAMES = [
    "travel_guide", "celebrity_info", "shopping_intent",
    "sentiment_label", "weather_info", "schedule_reminder", "translation_assist",
]

MODEL_PATH = Path("data/classifiers/stage1_model.pkl")


def load_dataset(seed: int = 42):
    """Load all test data. Returns (X, y) where y=1 means need_function."""
    data_dir = Path("data/test")
    all_samples: list[tuple[str, int]] = []

    for func in FUNCTION_NAMES:
        records = json.loads((data_dir / f"{func}.json").read_text())
        for r in records:
            last_user_msg = next(
                (m["content"] for m in reversed(r["conversation"]) if m["role"] == "user"), ""
            )
            all_samples.append((last_user_msg, 1))

    nf_records = json.loads((data_dir / "no_function.json").read_text())
    for r in nf_records:
        last_user_msg = next(
            (m["content"] for m in reversed(r["conversation"]) if m["role"] == "user"), ""
        )
        all_samples.append((last_user_msg, 0))

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(all_samples))
    all_samples = [all_samples[i] for i in idx]

    X = [s[0] for s in all_samples]
    y = [s[1] for s in all_samples]
    return X, y


def train(seed: int = 42):
    print("Loading dataset...")
    X, y = load_dataset(seed=seed)
    print(f"  Total: {len(X)} samples (need_function={sum(y)}, no_function={len(y)-sum(y)})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    print(f"  Train: {len(X_train)} / Test: {len(X_test)}")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=20000,
            sublinear_tf=True,
        )),
        ("clf", LinearSVC(C=1.0, max_iter=2000, random_state=seed)),
    ])

    print("\nTraining TF-IDF + LinearSVC...")
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    print(f"  Train time: {train_time:.2f}s")

    # Evaluate on held-out test set
    t1 = time.perf_counter()
    preds = pipeline.predict(X_test)
    infer_time = (time.perf_counter() - t1) / len(X_test) * 1000

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    print(f"\nHeld-out test metrics (n={len(X_test)}):")
    print(f"  Accuracy : {acc*100:.1f}%")
    print(f"  Precision: {prec*100:.1f}%")
    print(f"  Recall   : {rec*100:.1f}%")
    print(f"  F1       : {f1*100:.1f}%")
    print(f"  Latency  : {infer_time:.3f} ms/sample")

    # Re-train on full dataset for production model
    print("\nRetraining on full dataset for production...")
    t2 = time.perf_counter()
    pipeline.fit(X, y)
    retrain_time = time.perf_counter() - t2
    print(f"  Retrain time: {retrain_time:.2f}s")

    # Save
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nModel saved to: {MODEL_PATH}  ({MODEL_PATH.stat().st_size / 1024:.0f} KB)")

    return pipeline, {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "train_time_s": train_time, "latency_ms": infer_time,
        "n_train": len(X_train), "n_test": len(X_test),
    }


if __name__ == "__main__":
    train()
