"""Benchmark various no_function classifiers.

Compares:
1. Keyword-based baseline (current implementation)
2. TF-IDF + Logistic Regression
3. TF-IDF + SVM (Linear)
4. TF-IDF + Naive Bayes (MultinomialNB)
5. LLM-based (Ollama gemma3:4b, zero-shot)

All classifiers solve the binary task:
  need_function (True) vs no_function (False)

Metrics: Accuracy, Precision, Recall, F1, Latency
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass
from collections import Counter
from typing import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from src.classifier import FunctionCallClassifier


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

FUNCTION_NAMES = [
    "travel_guide", "celebrity_info", "shopping_intent",
    "sentiment_label", "weather_info", "schedule_reminder", "translation_assist"
]


def load_dataset(test_ratio: float = 0.2, seed: int = 42):
    """Load all test data and return (X_train, X_test, y_train, y_test).

    X: last user message (str)
    y: 1 = need_function, 0 = no_function
    """
    data_dir = Path("data/test")
    all_samples: list[tuple[str, int]] = []

    for func in FUNCTION_NAMES:
        path = data_dir / f"{func}.json"
        records = json.loads(path.read_text())
        for r in records:
            # Extract last user message
            last_user_msg = ""
            for msg in reversed(r["conversation"]):
                if msg["role"] == "user":
                    last_user_msg = msg["content"]
                    break
            all_samples.append((last_user_msg, 1))

    # no_function
    nf_path = data_dir / "no_function.json"
    nf_records = json.loads(nf_path.read_text())
    for r in nf_records:
        last_user_msg = ""
        for msg in reversed(r["conversation"]):
            if msg["role"] == "user":
                last_user_msg = msg["content"]
                break
        all_samples.append((last_user_msg, 0))

    # Shuffle
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(all_samples))
    all_samples = [all_samples[i] for i in idx]

    X = [s[0] for s in all_samples]
    y = [s[1] for s in all_samples]

    return train_test_split(X, y, test_size=test_ratio, random_state=seed, stratify=y)


# ──────────────────────────────────────────────
# Benchmark result
# ──────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    latency_ms: float       # avg per sample (inference only)
    train_time_s: float     # training time (0 for rule-based / LLM)
    tp: int
    fp: int
    fn: int
    tn: int
    notes: str = ""

    def print(self):
        print(f"\n{'='*60}")
        print(f"  {self.name}")
        print(f"{'='*60}")
        print(f"  Accuracy : {self.accuracy*100:.1f}%")
        print(f"  Precision: {self.precision*100:.1f}%")
        print(f"  Recall   : {self.recall*100:.1f}%")
        print(f"  F1       : {self.f1*100:.1f}%")
        print(f"  Latency  : {self.latency_ms:.3f} ms/sample")
        if self.train_time_s > 0:
            print(f"  Train    : {self.train_time_s:.2f} s")
        print(f"  TP={self.tp} FP={self.fp} FN={self.fn} TN={self.tn}")
        if self.notes:
            print(f"  Notes    : {self.notes}")


def compute_metrics(name, y_true, y_pred, latency_ms, train_time_s=0.0, notes=""):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, fp = cm[0]
    fn, tn = cm[1]
    return BenchmarkResult(
        name=name,
        accuracy=acc, precision=prec, recall=rec, f1=f1,
        latency_ms=latency_ms, train_time_s=train_time_s,
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        notes=notes,
    )


# ──────────────────────────────────────────────
# 1. Keyword baseline
# ──────────────────────────────────────────────

def run_keyword_baseline(X_test, y_test) -> BenchmarkResult:
    clf = FunctionCallClassifier()

    preds = []
    t0 = time.perf_counter()
    for text in X_test:
        result = clf.classify([{"role": "user", "content": text}])
        preds.append(1 if result.need_function else 0)
    elapsed = time.perf_counter() - t0

    latency_ms = elapsed / len(X_test) * 1000
    return compute_metrics(
        "1. Keyword Baseline (current)",
        y_test, preds, latency_ms,
        notes="ルールベース, 学習不要",
    )


# ──────────────────────────────────────────────
# 2. TF-IDF + Logistic Regression
# ──────────────────────────────────────────────

def run_tfidf_lr(X_train, X_test, y_train, y_test) -> BenchmarkResult:
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4),
            max_features=20000, sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            C=1.0, max_iter=1000, random_state=42,
        )),
    ])

    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    preds = pipeline.predict(X_test)
    latency_ms = (time.perf_counter() - t1) / len(X_test) * 1000

    return compute_metrics(
        "2. TF-IDF + Logistic Regression",
        y_test, preds, latency_ms, train_time,
        notes="char n-gram (2-4), max_features=20k",
    )


# ──────────────────────────────────────────────
# 3. TF-IDF + LinearSVC
# ──────────────────────────────────────────────

def run_tfidf_svm(X_train, X_test, y_train, y_test) -> BenchmarkResult:
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4),
            max_features=20000, sublinear_tf=True,
        )),
        ("clf", LinearSVC(C=1.0, max_iter=2000, random_state=42)),
    ])

    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    preds = pipeline.predict(X_test)
    latency_ms = (time.perf_counter() - t1) / len(X_test) * 1000

    return compute_metrics(
        "3. TF-IDF + LinearSVC",
        y_test, preds, latency_ms, train_time,
        notes="char n-gram (2-4), max_features=20k",
    )


# ──────────────────────────────────────────────
# 4. TF-IDF + Complement Naive Bayes
# ──────────────────────────────────────────────

def run_tfidf_nb(X_train, X_test, y_train, y_test) -> BenchmarkResult:
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4),
            max_features=20000, sublinear_tf=True,
        )),
        ("clf", ComplementNB(alpha=0.1)),
    ])

    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    preds = pipeline.predict(X_test)
    latency_ms = (time.perf_counter() - t1) / len(X_test) * 1000

    return compute_metrics(
        "4. TF-IDF + Complement NaiveBayes",
        y_test, preds, latency_ms, train_time,
        notes="char n-gram (2-4), CNB (suited for imbalanced)",
    )


# ──────────────────────────────────────────────
# 5. LLM (Ollama gemma3:4b, zero-shot) — sample only (expensive)
# ──────────────────────────────────────────────

LLM_SAMPLE_SIZE = 50  # Too slow to run all 818 samples

LLM_PROMPT_TEMPLATE = """あなたはFunction Routerです。
以下のユーザー発話に対して、外部APIや特定機能（Function Call）が必要かどうかを判断してください。

利用可能なFunction:
- travel_guide: 旅行・観光案内
- celebrity_info: 有名人・著名人情報
- shopping_intent: 購買意図・商品情報
- sentiment_label: 感情ラベル分析
- weather_info: 天気情報
- schedule_reminder: スケジュール・リマインダー
- translation_assist: 翻訳支援

ユーザー発話: {text}

上記のFunctionが必要なら "YES"、不要（雑談・一般的な質問等）なら "NO" とだけ答えてください。"""


def run_llm_classifier(X_test, y_test, sample_size: int = LLM_SAMPLE_SIZE) -> BenchmarkResult:
    try:
        import urllib.request
        import urllib.error
    except ImportError:
        pass

    # Sample balanced subset
    rng = np.random.default_rng(99)
    pos_idx = [i for i, y in enumerate(y_test) if y == 1]
    neg_idx = [i for i, y in enumerate(y_test) if y == 0]
    half = sample_size // 2
    chosen = (
        list(rng.choice(pos_idx, min(half, len(pos_idx)), replace=False)) +
        list(rng.choice(neg_idx, min(half, len(neg_idx)), replace=False))
    )
    chosen = sorted(chosen)

    X_sample = [X_test[i] for i in chosen]
    y_sample = [y_test[i] for i in chosen]

    preds = []
    total_time = 0.0
    errors = 0

    for text in X_sample:
        prompt = LLM_PROMPT_TEMPLATE.format(text=text)
        payload = json.dumps({
            "model": "gemma3:4b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_ctx": 512},
        }).encode()

        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read())
            elapsed = time.perf_counter() - t0
            total_time += elapsed

            response_text = body.get("response", "").strip().upper()
            preds.append(1 if "YES" in response_text else 0)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            total_time += elapsed
            errors += 1
            preds.append(0)  # fallback

    latency_ms = total_time / len(X_sample) * 1000

    notes = f"gemma3:4b zero-shot, n={len(X_sample)}"
    if errors:
        notes += f", {errors} errors"

    return compute_metrics(
        "5. LLM (gemma3:4b, zero-shot)",
        y_sample, preds, latency_ms,
        notes=notes,
    )


# ──────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────

def print_summary(results: list[BenchmarkResult]):
    print("\n")
    print("=" * 90)
    print("  BENCHMARK SUMMARY")
    print("=" * 90)
    header = f"{'Classifier':<38} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'ms/sample':>10} {'Train(s)':>9}"
    print(header)
    print("-" * 90)
    for r in results:
        train_str = f"{r.train_time_s:.2f}" if r.train_time_s > 0 else "   n/a"
        print(
            f"{r.name:<38} {r.accuracy*100:>5.1f}% {r.precision*100:>5.1f}% "
            f"{r.recall*100:>5.1f}% {r.f1*100:>5.1f}% "
            f"{r.latency_ms:>9.3f}  {train_str:>8}"
        )
    print("=" * 90)


def save_results(results: list[BenchmarkResult], out_path: Path):
    data = []
    for r in results:
        data.append({
            "name": r.name,
            "accuracy": round(r.accuracy, 4),
            "precision": round(r.precision, 4),
            "recall": round(r.recall, 4),
            "f1": round(r.f1, 4),
            "latency_ms": round(r.latency_ms, 4),
            "train_time_s": round(r.train_time_s, 4),
            "tp": r.tp, "fp": r.fp, "fn": r.fn, "tn": r.tn,
            "notes": r.notes,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"\nResults saved to: {out_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("Loading dataset...")
    X_train, X_test, y_train, y_test = load_dataset(test_ratio=0.2)

    label_dist = Counter(y_test)
    print(f"Test set: {len(y_test)} samples  "
          f"(need_function={label_dist[1]}, no_function={label_dist[0]})")

    results: list[BenchmarkResult] = []

    # 1. Keyword baseline
    print("\n[1/5] Keyword Baseline...")
    results.append(run_keyword_baseline(X_test, y_test))
    results[-1].print()

    # 2. TF-IDF + LR
    print("\n[2/5] TF-IDF + Logistic Regression...")
    results.append(run_tfidf_lr(X_train, X_test, y_train, y_test))
    results[-1].print()

    # 3. TF-IDF + SVM
    print("\n[3/5] TF-IDF + LinearSVC...")
    results.append(run_tfidf_svm(X_train, X_test, y_train, y_test))
    results[-1].print()

    # 4. TF-IDF + NB
    print("\n[4/5] TF-IDF + Complement NaiveBayes...")
    results.append(run_tfidf_nb(X_train, X_test, y_train, y_test))
    results[-1].print()

    # 5. LLM
    print(f"\n[5/5] LLM (gemma3:4b, zero-shot, n={LLM_SAMPLE_SIZE})...")
    print("  ※ Ollamaが起動していない場合はスキップされます")
    results.append(run_llm_classifier(X_test, y_test))
    results[-1].print()

    # Summary
    print_summary(results)

    # Save
    save_results(results, Path("data/results/classifier_benchmark.json"))


if __name__ == "__main__":
    main()
