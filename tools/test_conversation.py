"""Test conversation system with diverse inputs.

Runs 100 test conversations and analyzes function calling behavior.
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

from src.classifier import FunctionCallClassifier
from src.functions import init_all, registry


@dataclass
class TestCase:
    """Single test case."""
    input_text: str
    category: str  # expected behavior category
    expected_function: Optional[str] = None  # None means no_function expected


@dataclass
class TestResult:
    """Result of a single test."""
    input_text: str
    category: str
    expected_function: Optional[str]
    predicted_function: Optional[str]
    need_function: bool
    confidence: float
    is_correct: bool


# Test cases covering various scenarios
TEST_CASES = [
    # === 挨拶・日常会話 (no_function expected) ===
    TestCase("こんにちは", "greeting", None),
    TestCase("おはようございます", "greeting", None),
    TestCase("こんばんは", "greeting", None),
    TestCase("お疲れ様です", "greeting", None),
    TestCase("ありがとうございます", "greeting", None),
    TestCase("よろしくお願いします", "greeting", None),
    TestCase("さようなら", "greeting", None),
    TestCase("お元気ですか？", "greeting", None),
    TestCase("久しぶりですね", "greeting", None),
    TestCase("今日もよろしく", "greeting", None),

    # === 一般的な質問 (no_function expected) ===
    TestCase("AIとは何ですか？", "general_question", None),
    TestCase("プログラミングについて教えてください", "general_question", None),
    TestCase("数学の問題を解いてください", "general_question", None),
    TestCase("歴史について教えて", "general_question", None),
    TestCase("哲学とは何か説明して", "general_question", None),
    TestCase("英語の文法を教えて", "general_question", None),
    TestCase("物理学の基本を教えて", "general_question", None),
    TestCase("化学反応について", "general_question", None),
    TestCase("生物学の基礎", "general_question", None),
    TestCase("経済学入門", "general_question", None),

    # === 創作・アイデア (no_function expected) ===
    TestCase("物語を書いてください", "creative", None),
    TestCase("詩を作ってください", "creative", None),
    TestCase("アイデアを考えて", "creative", None),
    TestCase("キャッチコピーを作って", "creative", None),
    TestCase("小説の続きを書いて", "creative", None),
    TestCase("俳句を詠んで", "creative", None),
    TestCase("歌詞を書いて", "creative", None),
    TestCase("ストーリーを考えて", "creative", None),
    TestCase("プレゼンの構成を考えて", "creative", None),
    TestCase("名前を考えてください", "creative", None),

    # === 相談・意見 (no_function expected) ===
    TestCase("どう思いますか？", "opinion", None),
    TestCase("アドバイスをください", "opinion", None),
    TestCase("相談があります", "opinion", None),
    TestCase("悩みを聞いてください", "opinion", None),
    TestCase("助けてください", "opinion", None),
    TestCase("困っています", "opinion", None),
    TestCase("どうすればいいですか", "opinion", None),
    TestCase("おすすめは何ですか", "opinion", None),
    TestCase("選び方を教えて", "opinion", None),
    TestCase("比較してください", "opinion", None),

    # === 旅行関連 (travel_guide expected) ===
    TestCase("京都の観光名所を教えて", "travel", "travel_guide"),
    TestCase("東京でおすすめの場所は？", "travel", "travel_guide"),
    TestCase("大阪の名物を知りたい", "travel", "travel_guide"),
    TestCase("北海道の観光スポット", "travel", "travel_guide"),
    TestCase("沖縄旅行のプラン", "travel", "travel_guide"),
    TestCase("奈良に行きたい", "travel", "travel_guide"),
    TestCase("広島の見どころ", "travel", "travel_guide"),
    TestCase("福岡の観光案内", "travel", "travel_guide"),
    TestCase("名古屋城について", "travel", "travel_guide"),
    TestCase("横浜の観光情報", "travel", "travel_guide"),

    # === 天気関連 (weather_info expected) ===
    TestCase("明日の天気を教えて", "weather", "weather_info"),
    TestCase("東京の気温は？", "weather", "weather_info"),
    TestCase("週末は雨ですか", "weather", "weather_info"),
    TestCase("今日の天気予報", "weather", "weather_info"),
    TestCase("来週の天気", "weather", "weather_info"),
    TestCase("大阪は晴れますか", "weather", "weather_info"),
    TestCase("気温を教えて", "weather", "weather_info"),
    TestCase("傘は必要ですか", "weather", "weather_info"),
    TestCase("梅雨はいつまで", "weather", "weather_info"),
    TestCase("台風の情報", "weather", "weather_info"),

    # === 翻訳関連 (translation_assist expected) ===
    TestCase("これを英語にしてください", "translation", "translation_assist"),
    TestCase("日本語に翻訳して", "translation", "translation_assist"),
    TestCase("フランス語に訳して", "translation", "translation_assist"),
    TestCase("中国語で何と言いますか", "translation", "translation_assist"),
    TestCase("韓国語に変換して", "translation", "translation_assist"),
    TestCase("ドイツ語訳をお願い", "translation", "translation_assist"),
    TestCase("スペイン語にして", "translation", "translation_assist"),
    TestCase("イタリア語で", "translation", "translation_assist"),
    TestCase("ポルトガル語に", "translation", "translation_assist"),
    TestCase("ロシア語で言うと", "translation", "translation_assist"),

    # === 有名人関連 (celebrity_info expected) ===
    TestCase("大谷翔平について教えて", "celebrity", "celebrity_info"),
    TestCase("イチローの経歴", "celebrity", "celebrity_info"),
    TestCase("羽生結弦のプロフィール", "celebrity", "celebrity_info"),
    TestCase("宮崎駿監督について", "celebrity", "celebrity_info"),
    TestCase("村上春樹の作品", "celebrity", "celebrity_info"),
    TestCase("スティーブ・ジョブズについて", "celebrity", "celebrity_info"),
    TestCase("ビル・ゲイツの経歴", "celebrity", "celebrity_info"),
    TestCase("アインシュタインについて", "celebrity", "celebrity_info"),
    TestCase("ピカソの作品", "celebrity", "celebrity_info"),
    TestCase("モーツァルトについて", "celebrity", "celebrity_info"),

    # === 感情分析 (sentiment_label expected) ===
    TestCase("この文章の感情を分析して", "sentiment", "sentiment_label"),
    TestCase("ポジティブかネガティブか判定して", "sentiment", "sentiment_label"),
    TestCase("感情ラベルをつけて", "sentiment", "sentiment_label"),
    TestCase("気持ちを分析して", "sentiment", "sentiment_label"),
    TestCase("センチメント分析をお願い", "sentiment", "sentiment_label"),

    # === スケジュール関連 (schedule_reminder expected) ===
    TestCase("明日の予定をリマインドして", "schedule", "schedule_reminder"),
    TestCase("会議を設定して", "schedule", "schedule_reminder"),
    TestCase("スケジュールを確認", "schedule", "schedule_reminder"),
    TestCase("リマインダーを設定", "schedule", "schedule_reminder"),
    TestCase("予約を入れて", "schedule", "schedule_reminder"),

    # === 買い物関連 (shopping_intent expected) ===
    TestCase("おすすめの商品を教えて", "shopping", "shopping_intent"),
    TestCase("この商品を購入したい", "shopping", "shopping_intent"),
    TestCase("価格を比較して", "shopping", "shopping_intent"),
    TestCase("セール情報を教えて", "shopping", "shopping_intent"),
    TestCase("レビューを見せて", "shopping", "shopping_intent"),
]


def run_test():
    """Run all test cases through the classifier."""
    # Initialize
    init_all()
    classifier = FunctionCallClassifier()

    results: list[TestResult] = []

    print("=" * 60)
    print("FuncGemma Classifier Test (100 cases)")
    print("=" * 60)
    print()

    for i, tc in enumerate(TEST_CASES):
        # Create conversation
        conversation = [{"role": "user", "content": tc.input_text}]

        # Classify
        classification = classifier.classify(conversation)

        # Determine predicted function
        if classification.need_function:
            predicted = classification.matched_function
        else:
            predicted = None

        # Check correctness
        is_correct = (predicted == tc.expected_function)

        result = TestResult(
            input_text=tc.input_text,
            category=tc.category,
            expected_function=tc.expected_function,
            predicted_function=predicted,
            need_function=classification.need_function,
            confidence=classification.confidence,
            is_correct=is_correct
        )
        results.append(result)

        # Print progress
        status = "✓" if is_correct else "✗"
        print(f"[{i+1:3d}] {status} {tc.input_text[:30]:<30} | "
              f"expected: {tc.expected_function or 'no_function':<20} | "
              f"predicted: {predicted or 'no_function':<20}")

    # Analyze results
    print()
    print("=" * 60)
    print("Results Analysis")
    print("=" * 60)

    # Overall accuracy
    correct_count = sum(1 for r in results if r.is_correct)
    total = len(results)
    print(f"\nOverall Accuracy: {correct_count}/{total} ({correct_count/total*100:.1f}%)")

    # Per-category analysis
    print("\nPer-Category Results:")
    print("-" * 50)

    category_results = defaultdict(lambda: {"correct": 0, "total": 0, "errors": []})
    for r in results:
        cat = r.category
        category_results[cat]["total"] += 1
        if r.is_correct:
            category_results[cat]["correct"] += 1
        else:
            category_results[cat]["errors"].append(r)

    for cat, data in sorted(category_results.items()):
        acc = data["correct"] / data["total"] * 100
        print(f"  {cat:<20}: {data['correct']:2d}/{data['total']:2d} ({acc:5.1f}%)")

    # Error analysis
    errors = [r for r in results if not r.is_correct]
    if errors:
        print(f"\nErrors ({len(errors)} cases):")
        print("-" * 50)
        for e in errors:
            print(f"  Input: {e.input_text}")
            print(f"    Expected: {e.expected_function or 'no_function'}")
            print(f"    Got:      {e.predicted_function or 'no_function'}")
            print(f"    Confidence: {e.confidence:.2f}")
            print()

    # Function distribution
    print("\nPrediction Distribution:")
    print("-" * 50)
    func_counts = defaultdict(int)
    for r in results:
        func_counts[r.predicted_function or "no_function"] += 1

    for func, count in sorted(func_counts.items(), key=lambda x: -x[1]):
        print(f"  {func:<20}: {count:3d}")

    # Confusion matrix style summary
    print("\nConfusion Summary (Expected vs Predicted):")
    print("-" * 50)

    confusion = defaultdict(lambda: defaultdict(int))
    for r in results:
        exp = r.expected_function or "no_function"
        pred = r.predicted_function or "no_function"
        confusion[exp][pred] += 1

    all_funcs = sorted(set(
        [r.expected_function or "no_function" for r in results] +
        [r.predicted_function or "no_function" for r in results]
    ))

    # Print header
    print(f"{'':20}", end="")
    for f in all_funcs:
        print(f"{f[:8]:>10}", end="")
    print()

    for exp in all_funcs:
        print(f"{exp:<20}", end="")
        for pred in all_funcs:
            count = confusion[exp][pred]
            if count > 0:
                print(f"{count:>10}", end="")
            else:
                print(f"{'·':>10}", end="")
        print()

    # Save results
    output_file = Path("data/results/classifier_test_100.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "total": total,
        "correct": correct_count,
        "accuracy": correct_count / total,
        "per_category": {
            cat: {
                "correct": data["correct"],
                "total": data["total"],
                "accuracy": data["correct"] / data["total"]
            }
            for cat, data in category_results.items()
        },
        "errors": [
            {
                "input": e.input_text,
                "category": e.category,
                "expected": e.expected_function,
                "predicted": e.predicted_function,
                "confidence": e.confidence
            }
            for e in errors
        ]
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    run_test()
