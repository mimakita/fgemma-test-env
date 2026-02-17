"""Two-stage function calling prototype.

Stage 1: Classify if function call is needed (binary classification)
Stage 2: If needed, use FunctionGemma to select function

This approach separates "whether to call" from "what to call".

Usage:
    python -m tools.prototype_two_stage --run-id 5
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
FINETUNE_DIR = PROJECT_ROOT / "data" / "finetune"
PEFT_ADAPTERS_DIR = PROJECT_ROOT / "data" / "peft_adapters"

DEFAULT_MODEL = "google/functiongemma-270m-it"


# Stage 1: Simple heuristic-based classifier for function call necessity
# In production, this would be a separate trained model

# Keywords that suggest NO function call is needed
NO_FUNCTION_KEYWORDS = [
    # 一般的な挨拶・雑談
    "こんにちは", "おはよう", "こんばんは", "ありがとう", "すみません",
    "元気", "調子", "久しぶり", "よろしく", "はじめまして",
    "お疲れ様", "おやすみ", "さようなら", "またね",
    # 意見・感想を求める
    "どう思う", "どう思います", "意見を", "感想を",
    "思いますか", "考えますか", "でしょうか",
    # 一般知識・説明要求
    "とは何", "とは", "って何", "について教えて", "説明して",
    "なぜ", "どうして", "理由は", "原因は",
    "どういう意味", "違いは", "どう違う",
    # 創作・アイデア・相談
    "アイデア", "案を", "考えて", "提案", "作って",
    "書いて", "作成して", "ストーリー", "物語",
    # 数学・計算（関数なし）
    "計算", "足し算", "引き算", "掛け算", "割り算",
    "何倍", "パーセント",
    # プログラミング（関数なし）
    "コード", "プログラム", "バグ", "エラー", "関数",
    # 個人的な相談
    "悩み", "相談", "アドバイス", "困って",
    "どうすれば", "どうしたら", "迷って",
    # 雑談・日常会話
    "最近", "今日は", "週末", "休み", "趣味",
    "好き", "嫌い", "面白い", "つまらない",
    "美味しい", "きれい", "すごい",
    # 質問への回答
    "わかりました", "了解", "なるほど", "そうですね",
]

# Strong indicators of no function (higher weight)
NO_FUNCTION_STRONG = [
    "〜とは何ですか", "〜って何ですか", "教えてください",
    "どう思いますか", "どうすればいい",
    "アドバイスをください", "相談があります",
    "理由を教えて", "違いを教えて",
]

# Keywords that suggest function call IS needed
FUNCTION_KEYWORDS = {
    "travel_guide": ["旅行", "観光", "行き方", "名所", "ホテル", "おすすめの場所"],
    "celebrity_info": ["有名人", "芸能人", "歌手", "俳優", "選手", "について知りたい"],
    "shopping_intent": ["買いたい", "購入", "おすすめの商品", "どこで買える", "値段"],
    "sentiment_label": ["感情分析", "気持ち", "感情を分類", "ポジティブ", "ネガティブ"],
    "weather_info": ["天気", "気温", "雨", "晴れ", "予報", "明日の天気"],
    "schedule_reminder": ["予定", "スケジュール", "リマインダー", "忘れないように", "時に通知"],
    "translation_assist": ["翻訳", "英語にして", "日本語にして", "英訳", "和訳", "〜語で"],
}


def classify_need_function(conversation: list[dict]) -> tuple[bool, float]:
    """Stage 1: Classify if function call is needed.

    Returns:
        (need_function: bool, confidence: float)
    """
    # Get last user message
    last_user_msg = ""
    for msg in reversed(conversation):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break

    # Combine all user messages for context
    all_user_text = " ".join(
        msg.get("content", "") for msg in conversation if msg.get("role") == "user"
    )

    # Count no_function keywords
    no_func_score = 0
    for kw in NO_FUNCTION_KEYWORDS:
        if kw in all_user_text:
            no_func_score += 1

    # Check strong no_function indicators
    for strong in NO_FUNCTION_STRONG:
        pattern = strong.replace("〜", "")
        if pattern in all_user_text:
            no_func_score += 3  # Higher weight

    # Count function keywords (only in last user message for precision)
    func_score = 0
    matched_functions = []
    for func_name, keywords in FUNCTION_KEYWORDS.items():
        for kw in keywords:
            if kw in last_user_msg:
                func_score += 2  # Last message is more important
                matched_functions.append(func_name)
                break
            elif kw in all_user_text:
                func_score += 1
                matched_functions.append(func_name)
                break

    # Decision logic
    # If strong function signal in last message, likely need function
    if func_score >= 3:
        return True, min(0.6 + func_score * 0.1, 0.95)

    # If no function keywords and some no_function keywords
    if func_score == 0 and no_func_score >= 2:
        return False, min(0.5 + no_func_score * 0.05, 0.85)

    # If function score is higher than no_function
    if func_score > no_func_score:
        return True, 0.5 + (func_score - no_func_score) * 0.1

    # If no_function score is significantly higher
    if no_func_score > func_score + 2:
        return False, 0.5 + (no_func_score - func_score) * 0.05

    # Ambiguous case - if any function keyword, call function
    if func_score > 0:
        return True, 0.5

    # Default: no function (more conservative)
    if no_func_score > 0:
        return False, 0.4

    # No signals either way - default to function (safer)
    return True, 0.3


def load_tools_from_training_data(run_dir: Path) -> list[dict]:
    """Load tool schemas from training data."""
    train_file = run_dir / "train.jsonl"
    with open(train_file, "r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())
        return first_line.get("tools", [])


def format_prompt(messages: list[dict], tools: list[dict]) -> str:
    """Format messages and tools into FunctionGemma prompt format."""
    parts = []

    tools_text = "Available tools:\n"
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "")
        desc = func.get("description", "")
        tools_text += f"- {name}: {desc}\n"
    parts.append(tools_text)

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
        elif role == "assistant":
            parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")

    parts.append("<start_of_turn>model\n")
    return "\n".join(parts)


def extract_function_from_response(response: str, function_names: list[str]) -> str:
    """Extract function name from model response."""
    try:
        data = json.loads(response.strip())
        if isinstance(data, dict) and "name" in data:
            return data["name"]
    except:
        pass

    for func_name in function_names:
        if func_name in response:
            return func_name

    return "no_function"


def main():
    parser = argparse.ArgumentParser(description="Two-stage function calling prototype")
    parser.add_argument("--run-id", type=int, required=True, help="Run ID")
    parser.add_argument("--checkpoint", type=int, default=800, help="Checkpoint step")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base model")
    parser.add_argument("--stage1-only", action="store_true", help="Only run Stage 1")
    args = parser.parse_args()

    run_dir = FINETUNE_DIR / f"run_{args.run_id}"
    adapter_dir = PEFT_ADAPTERS_DIR / f"run_{args.run_id}" / f"checkpoint-{args.checkpoint}"

    # Load test data
    test_file = run_dir / "all_test_data.json"
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    logger.info(f"Loaded {len(test_data)} test samples")

    # Load tools
    tools = load_tools_from_training_data(run_dir)
    function_names = [t["function"]["name"] for t in tools]
    logger.info(f"Functions: {function_names}")

    # Stage 1 only mode
    if args.stage1_only:
        logger.info("Running Stage 1 classification only...")
        stage1_results = {"correct": 0, "total": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0}

        for sample in test_data:
            messages = sample.get("conversation", sample.get("messages", []))
            expected = sample.get("expected_function", "no_function")
            actual_needs_func = expected != "no_function" and expected is not None

            predicted_needs_func, confidence = classify_need_function(messages)

            if predicted_needs_func == actual_needs_func:
                stage1_results["correct"] += 1
                if actual_needs_func:
                    stage1_results["tp"] += 1
                else:
                    stage1_results["tn"] += 1
            else:
                if predicted_needs_func:
                    stage1_results["fp"] += 1
                else:
                    stage1_results["fn"] += 1

            stage1_results["total"] += 1

        acc = stage1_results["correct"] / stage1_results["total"] * 100
        precision = stage1_results["tp"] / (stage1_results["tp"] + stage1_results["fp"]) * 100 if (stage1_results["tp"] + stage1_results["fp"]) > 0 else 0
        recall = stage1_results["tp"] / (stage1_results["tp"] + stage1_results["fn"]) * 100 if (stage1_results["tp"] + stage1_results["fn"]) > 0 else 0

        print("\n" + "=" * 60)
        print("Stage 1 Classification Results (Heuristic)")
        print("=" * 60)
        print(f"Accuracy: {acc:.1f}%")
        print(f"Precision (need_func): {precision:.1f}%")
        print(f"Recall (need_func): {recall:.1f}%")
        print(f"True Positives: {stage1_results['tp']}")
        print(f"True Negatives (no_function detected): {stage1_results['tn']}")
        print(f"False Positives: {stage1_results['fp']}")
        print(f"False Negatives: {stage1_results['fn']}")
        print("=" * 60)
        return

    # Full two-stage evaluation
    logger.info("Loading model for Stage 2...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    results_by_func = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "total": 0})
    correct = 0
    total = 0
    stage1_blocked = 0

    logger.info("Starting two-stage evaluation...")
    for i, sample in enumerate(test_data):
        messages = sample.get("conversation", sample.get("messages", []))
        expected = sample.get("expected_function", "no_function")
        if expected is None:
            expected = "no_function"

        # Stage 1: Check if function call is needed
        need_func, confidence = classify_need_function(messages)

        if not need_func:
            # Stage 1 says no function needed
            predicted = "no_function"
            stage1_blocked += 1
        else:
            # Stage 2: Use FunctionGemma to select function
            prompt = format_prompt(messages, tools)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            predicted = extract_function_from_response(response, function_names)

        results_by_func[expected]["total"] += 1

        if predicted == expected:
            correct += 1
            results_by_func[expected]["tp"] += 1
        elif expected != "no_function" and predicted != "no_function":
            results_by_func[expected]["fn"] += 1
            results_by_func[predicted]["fp"] += 1
        elif expected != "no_function":
            results_by_func[expected]["fn"] += 1
        elif predicted != "no_function":
            results_by_func[predicted]["fp"] += 1

        total += 1

        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i+1}/{len(test_data)}, Accuracy: {correct/total*100:.1f}%")

    # Print results
    print("\n" + "=" * 70)
    print(f"Two-Stage Evaluation Results (Run {args.run_id}, checkpoint-{args.checkpoint})")
    print("=" * 70)
    print(f"\nOverall Accuracy: {correct/total*100:.1f}% ({correct}/{total})")
    print(f"Stage 1 blocked: {stage1_blocked} samples")

    print(f"\n{'Function':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Total':>8}")
    print("-" * 70)

    for func_name in function_names + ["no_function"]:
        stats = results_by_func[func_name]
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        total_func = stats["total"]

        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{func_name:<25} {precision:>9.1f}% {recall:>9.1f}% {f1:>9.1f}% {total_func:>8}")

    print("=" * 70)


if __name__ == "__main__":
    main()
