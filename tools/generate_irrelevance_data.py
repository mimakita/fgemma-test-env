"""Generate irrelevance-augmented data for no_function detection.

Based on research findings (arxiv:2412.01130v2, arxiv:2410.04587v2):
- Create negative samples by removing relevant functions from tool list
- Train model to recognize when NO available function matches the query

This approach addresses the "inverse relationship" problem where
fine-tuning for function calling accuracy degrades irrelevance detection.

Usage:
    python -m tools.generate_irrelevance_data --count 500
"""

import argparse
import json
import logging
import random
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "test"

FUNCTION_NAMES = [
    "travel_guide",
    "celebrity_info",
    "shopping_intent",
    "sentiment_label",
    "weather_info",
    "schedule_reminder",
    "translation_assist",
]


def load_function_data() -> dict[str, list[dict]]:
    """Load all function test data."""
    data = {}
    for func_name in FUNCTION_NAMES:
        filepath = DATA_DIR / f"{func_name}.json"
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                data[func_name] = json.load(f)
            logger.info(f"Loaded {len(data[func_name])} cases for {func_name}")
    return data


def generate_irrelevance_cases(
    function_data: dict[str, list[dict]],
    count: int,
    seed: int = 42,
) -> list[dict]:
    """Generate irrelevance-augmented cases.

    Strategy: Take existing function-call queries and mark them as no_function
    because the correct function is "not available" in the tool list.

    This trains the model to recognize that even if a query LOOKS like
    it needs a function, it should not call one if no matching function exists.
    """
    rng = random.Random(seed)

    # Flatten all function data
    all_cases = []
    for func_name, cases in function_data.items():
        for case in cases:
            all_cases.append({
                "original_function": func_name,
                "case": case,
            })

    rng.shuffle(all_cases)

    generated = []
    for i in range(min(count, len(all_cases))):
        original = all_cases[i % len(all_cases)]
        func_name = original["original_function"]
        case = original["case"]

        # Create a no_function version
        # The conversation looks like it needs a function, but it's marked as no_function
        # because we'll exclude the correct function from the tool list during training
        new_case = {
            "conversation": case["conversation"],
            "expected_function": None,
            "expected_arguments": None,
            "test_id": f"irrelevance_{i:04d}",
            "category": "no_function",
            "original_function": func_name,  # For reference/debugging
            "irrelevance_type": "function_excluded",
        }
        generated.append(new_case)

    return generated


def generate_ambiguous_cases(count: int, seed: int = 42) -> list[dict]:
    """Generate cases that are ambiguous and shouldn't trigger functions.

    These are edge cases where the query mentions concepts related to functions
    but doesn't actually request the function's action.
    """
    rng = random.Random(seed)

    templates = [
        # Weather-related but not forecast request
        {
            "conversation": [
                {"role": "user", "content": "天気予報を見る習慣がなくて、よく傘を忘れます。"},
                {"role": "assistant", "content": "天気予報アプリを使うと便利ですよ。"},
                {"role": "user", "content": "そうですね、試してみます。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "天気の話って会話のきっかけになりやすいですよね。"},
            ],
        },
        # Travel-related but not guide request
        {
            "conversation": [
                {"role": "user", "content": "旅行の計画を立てるのは楽しいですよね。"},
                {"role": "assistant", "content": "どこか行きたい場所はありますか？"},
                {"role": "user", "content": "まだ決めてないんです。のんびり考えます。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "飛行機の座席は窓側派ですか？通路側派ですか？"},
            ],
        },
        # Celebrity-related but not info request
        {
            "conversation": [
                {"role": "user", "content": "最近、テレビをあまり見なくなりました。"},
                {"role": "assistant", "content": "動画配信サービスを使っていますか？"},
                {"role": "user", "content": "はい、そっちの方が便利なので。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "有名人になりたいと思ったことはありますか？"},
            ],
        },
        # Translation-related but not translation request
        {
            "conversation": [
                {"role": "user", "content": "外国語を学ぶのに一番良い方法って何だと思いますか？"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "翻訳家って大変な仕事だと思います。"},
                {"role": "assistant", "content": "言語能力だけでなく、文化理解も必要ですからね。"},
                {"role": "user", "content": "AIの進歩で仕事が減ってるのかな。"},
            ],
        },
        # Schedule-related but not reminder request
        {
            "conversation": [
                {"role": "user", "content": "時間管理って難しいですよね。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "会議が多すぎて疲れます。"},
                {"role": "assistant", "content": "リモートワークだと特に多いですよね。"},
                {"role": "user", "content": "そうなんです。効率化したいです。"},
            ],
        },
        # Shopping-related but not purchase intent
        {
            "conversation": [
                {"role": "user", "content": "最近の物価高、厳しいですよね。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "ネットショッピングとお店、どちらが好きですか？"},
                {"role": "assistant", "content": "便利さではネット、体験ではお店ですね。"},
                {"role": "user", "content": "確かに、一長一短ありますよね。"},
            ],
        },
        # Sentiment-related but not analysis request
        {
            "conversation": [
                {"role": "user", "content": "感情を言葉にするのって難しいですよね。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "SNSのコメント欄って感情的になりやすいですよね。"},
                {"role": "assistant", "content": "匿名性が影響しているかもしれませんね。"},
                {"role": "user", "content": "冷静に議論できる場があればいいのに。"},
            ],
        },
        # Multi-domain but no clear action
        {
            "conversation": [
                {"role": "user", "content": "AIって色々なことができるようになりましたね。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "最近のテクノロジーの進歩はすごいですね。"},
                {"role": "assistant", "content": "特にどの分野が気になりますか？"},
                {"role": "user", "content": "全般的に興味があります。"},
            ],
        },
    ]

    generated = []
    for i in range(count):
        template = rng.choice(templates)
        new_case = {
            "conversation": template["conversation"],
            "expected_function": None,
            "expected_arguments": None,
            "test_id": f"ambiguous_{i:04d}",
            "category": "no_function",
            "irrelevance_type": "ambiguous_query",
        }
        generated.append(new_case)

    return generated


def main():
    parser = argparse.ArgumentParser(
        description="Generate irrelevance-augmented data for no_function detection"
    )
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=500,
        help="Number of irrelevance cases to generate (default: 500)",
    )
    parser.add_argument(
        "--ambiguous", "-a",
        type=int,
        default=200,
        help="Number of ambiguous cases to generate (default: 200)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge with existing no_function.json",
    )
    args = parser.parse_args()

    # Load function data for irrelevance generation
    function_data = load_function_data()

    # Generate irrelevance cases
    logger.info(f"Generating {args.count} irrelevance-augmented cases...")
    irrelevance_cases = generate_irrelevance_cases(
        function_data, args.count, args.seed
    )

    # Generate ambiguous cases
    logger.info(f"Generating {args.ambiguous} ambiguous cases...")
    ambiguous_cases = generate_ambiguous_cases(args.ambiguous, args.seed)

    all_new_cases = irrelevance_cases + ambiguous_cases
    logger.info(f"Total new cases: {len(all_new_cases)}")

    # Merge with existing no_function data
    if args.merge:
        no_function_file = DATA_DIR / "no_function.json"
        if no_function_file.exists():
            with open(no_function_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            logger.info(f"Loaded {len(existing)} existing no_function cases")

            # Get max ID
            max_id = 0
            for case in existing:
                tid = case.get("test_id", "")
                if tid.startswith("no_function_"):
                    try:
                        num = int(tid.split("_")[-1])
                        max_id = max(max_id, num)
                    except ValueError:
                        pass

            # Renumber new cases
            for i, case in enumerate(all_new_cases):
                if case["test_id"].startswith("irrelevance_"):
                    case["test_id"] = f"no_function_{max_id + 1 + i:04d}"
                elif case["test_id"].startswith("ambiguous_"):
                    case["test_id"] = f"no_function_{max_id + 1 + i:04d}"

            all_cases = existing + all_new_cases
            logger.info(f"Total after merge: {len(all_cases)} cases")
        else:
            all_cases = all_new_cases
    else:
        all_cases = all_new_cases

    # Save
    output_file = DATA_DIR / "no_function.json" if args.merge else DATA_DIR / "no_function_irrelevance.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved to {output_file}")

    # Show distribution
    print("\n=== New Cases Distribution ===")
    irrelevance_count = len([c for c in all_new_cases if c.get("irrelevance_type") == "function_excluded"])
    ambiguous_count = len([c for c in all_new_cases if c.get("irrelevance_type") == "ambiguous_query"])
    print(f"  Function-excluded (irrelevance): {irrelevance_count}")
    print(f"  Ambiguous queries: {ambiguous_count}")

    if args.merge:
        print(f"\n  Total no_function: {len(all_cases)}")

    # Show samples
    print("\n=== Sample Generated Cases ===")
    rng = random.Random(args.seed)
    for i, case in enumerate(rng.sample(all_new_cases, min(5, len(all_new_cases)))):
        print(f"\n[{i+1}] {case.get('test_id')} ({case.get('irrelevance_type')})")
        for msg in case["conversation"][:2]:
            content = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
            print(f"  {msg['role']}: {content}")


if __name__ == "__main__":
    main()
