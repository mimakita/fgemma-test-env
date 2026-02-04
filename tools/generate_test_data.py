"""Test data generation tool using qwen2.5:7b via Ollama.

Generates conversation test data for evaluating FunctionGemma routing accuracy.
Each test case contains a multi-turn conversation and the expected function (or none).

Usage:
    python -m tools.generate_test_data [--function FUNC_NAME] [--count N] [--no-function-count N]
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import ollama

from src.config import TEST_DATA_MODEL, TEST_DATA_OPTIONS, TEST_DATA_PER_FUNCTION, TEST_DATA_NO_FUNCTION, TEST_DATA_BATCH_SIZE
from src.functions import init_all, registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "test"


def generate_function_prompt(func_name: str, description: str, parameters: dict, batch_size: int) -> str:
    """Create a prompt for generating test conversations for a specific function."""
    params_str = json.dumps(parameters, indent=2, ensure_ascii=False)
    return f"""あなたはテストデータ生成の専門家です。
以下の関数に対して、ユーザーの対話がこの関数を呼び出すべきケースを{batch_size}件生成してください。

## 関数情報
- 関数名: {func_name}
- 説明: {description}
- パラメータ:
{params_str}

## 出力形式
JSON配列で出力してください。各要素は以下の形式:

```json
[
  {{
    "conversation": [
      {{"role": "user", "content": "ユーザーの発言"}},
      {{"role": "assistant", "content": "アシスタントの応答"}},
      {{"role": "user", "content": "ユーザーの2回目の発言"}}
    ],
    "expected_function": "{func_name}",
    "expected_arguments": {{
      "パラメータ名": "値"
    }}
  }}
]
```

## 要件
- 各対話は2~4ターン（user + assistant の組み合わせ）
- 日本語と英語を混ぜて多様性を持たせる（7割日本語、3割英語）
- 直接的な表現と間接的な表現の両方を含める
- 様々なトピックとシチュエーションを使う
- expected_arguments は関数のパラメータに合った現実的な値を入れる
- JSON以外のテキストは出力しない
- 必ず{batch_size}件生成する"""


def generate_no_function_prompt(function_names: list[str], batch_size: int) -> str:
    """Create a prompt for generating conversations that should NOT trigger any function."""
    funcs_str = ", ".join(function_names)
    return f"""あなたはテストデータ生成の専門家です。
以下の関数のどれにも該当しない、通常の対話を{batch_size}件生成してください。

## 登録されている関数（これらを呼び出すべきでない対話を生成）
{funcs_str}

## 出力形式
JSON配列で出力してください。各要素は以下の形式:

```json
[
  {{
    "conversation": [
      {{"role": "user", "content": "ユーザーの発言"}},
      {{"role": "assistant", "content": "アシスタントの応答"}},
      {{"role": "user", "content": "ユーザーの2回目の発言"}}
    ],
    "expected_function": null,
    "expected_arguments": null
  }}
]
```

## 要件
- 各対話は2~4ターン
- 日本語と英語を混ぜる（7割日本語、3割英語）
- 以下のカテゴリから多様に生成:
  * 挨拶や雑談
  * 一般知識の質問
  * 意見交換やディスカッション
  * 数学や論理パズル
  * 創作や物語の依頼
  * 個人的な相談（非商業）
  * 技術的な質問（プログラミング等）
  * 日常的な話題
- どの関数にも明確に該当しないこと
- JSON以外のテキストは出力しない
- 必ず{batch_size}件生成する"""


def extract_json_from_response(text: str) -> Optional[list]:
    """Extract JSON array from LLM response, handling markdown code blocks."""
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return None
    except json.JSONDecodeError:
        pass

    # Try finding JSON array in text
    bracket_start = text.find('[')
    bracket_end = text.rfind(']')
    if bracket_start >= 0 and bracket_end > bracket_start:
        try:
            data = json.loads(text[bracket_start:bracket_end + 1])
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    return None


def validate_test_case(case: dict, func_name: Optional[str], all_func_names: list[str]) -> bool:
    """Validate a single test case."""
    if not isinstance(case, dict):
        return False

    # Must have conversation
    conv = case.get("conversation")
    if not isinstance(conv, list) or len(conv) < 1:
        return False

    # Each message must have role and content
    for msg in conv:
        if not isinstance(msg, dict):
            return False
        if "role" not in msg or "content" not in msg:
            return False
        if msg["role"] not in ("user", "assistant"):
            return False
        if not msg["content"] or not msg["content"].strip():
            return False

    # Must have at least one user message
    if not any(m["role"] == "user" for m in conv):
        return False

    # Validate expected function
    expected = case.get("expected_function")
    if func_name is not None:
        # Should match the target function
        if expected != func_name:
            return False
    else:
        # Should be null/None
        if expected is not None and expected not in (None, "null", "none", ""):
            return False

    return True


def generate_batch(
    prompt: str,
    func_name: Optional[str],
    all_func_names: list[str],
    batch_num: int,
    max_retries: int = 3,
) -> list[dict]:
    """Generate a batch of test data using the LLM."""
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=TEST_DATA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options=TEST_DATA_OPTIONS,
            )
            text = response.message.content or ""
            data = extract_json_from_response(text)

            if data is None:
                logger.warning(
                    f"  Batch {batch_num} attempt {attempt + 1}: Failed to parse JSON"
                )
                continue

            # Validate and filter
            valid = []
            for case in data:
                if validate_test_case(case, func_name, all_func_names):
                    # Normalize expected_function for no-function cases
                    if func_name is None:
                        case["expected_function"] = None
                        case["expected_arguments"] = None
                    valid.append(case)

            if valid:
                logger.info(
                    f"  Batch {batch_num}: {len(valid)}/{len(data)} valid cases"
                )
                return valid
            else:
                logger.warning(
                    f"  Batch {batch_num} attempt {attempt + 1}: No valid cases from {len(data)} generated"
                )

        except Exception as e:
            logger.error(f"  Batch {batch_num} attempt {attempt + 1} error: {e}")
            time.sleep(2)

    logger.error(f"  Batch {batch_num}: Failed after {max_retries} retries")
    return []


def generate_for_function(func_name: str, target_count: int) -> list[dict]:
    """Generate test data for a specific function."""
    func_def = registry.get(func_name)
    if not func_def:
        logger.error(f"Unknown function: {func_name}")
        return []

    all_func_names = registry.get_all_names()
    all_cases = []
    batch_num = 0

    while len(all_cases) < target_count:
        remaining = target_count - len(all_cases)
        batch_size = min(TEST_DATA_BATCH_SIZE, remaining)

        prompt = generate_function_prompt(
            func_name, func_def.description, func_def.parameters, batch_size
        )

        batch_num += 1
        cases = generate_batch(prompt, func_name, all_func_names, batch_num)
        all_cases.extend(cases)

        logger.info(f"  Total for {func_name}: {len(all_cases)}/{target_count}")

    return all_cases[:target_count]


def generate_no_function(target_count: int) -> list[dict]:
    """Generate test data that should not trigger any function."""
    all_func_names = registry.get_all_names()
    all_cases = []
    batch_num = 0

    while len(all_cases) < target_count:
        remaining = target_count - len(all_cases)
        batch_size = min(TEST_DATA_BATCH_SIZE, remaining)

        prompt = generate_no_function_prompt(all_func_names, batch_size)

        batch_num += 1
        cases = generate_batch(prompt, None, all_func_names, batch_num)
        all_cases.extend(cases)

        logger.info(f"  Total for no_function: {len(all_cases)}/{target_count}")

    return all_cases[:target_count]


def add_test_ids(cases: list[dict], prefix: str) -> list[dict]:
    """Add unique test IDs to each case."""
    for i, case in enumerate(cases):
        case["test_id"] = f"{prefix}_{i + 1:04d}"
        case["category"] = prefix
    return cases


def save_test_data(cases: list[dict], filename: str):
    """Save test data to JSON file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(cases)} cases to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Generate test data for FuncGemma evaluation")
    parser.add_argument(
        "--function", "-f",
        type=str,
        default=None,
        help="Generate data for a specific function only (default: all)",
    )
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=TEST_DATA_PER_FUNCTION,
        help=f"Number of test cases per function (default: {TEST_DATA_PER_FUNCTION})",
    )
    parser.add_argument(
        "--no-function-count", "-n",
        type=int,
        default=TEST_DATA_NO_FUNCTION,
        help=f"Number of no-function test cases (default: {TEST_DATA_NO_FUNCTION})",
    )
    parser.add_argument(
        "--skip-no-function",
        action="store_true",
        help="Skip generating no-function test data",
    )
    args = parser.parse_args()

    # Initialize function registry
    init_all()
    all_func_names = registry.get_all_names()
    logger.info(f"Registered functions: {all_func_names}")

    all_test_data = []

    # Generate function-specific test data
    if args.function:
        if args.function not in all_func_names:
            logger.error(f"Unknown function: {args.function}")
            logger.error(f"Available: {all_func_names}")
            sys.exit(1)
        functions_to_generate = [args.function]
    else:
        functions_to_generate = all_func_names

    for func_name in functions_to_generate:
        logger.info(f"\n=== Generating test data for: {func_name} ===")
        cases = generate_for_function(func_name, args.count)
        cases = add_test_ids(cases, func_name)
        save_test_data(cases, f"{func_name}.json")
        all_test_data.extend(cases)

    # Generate no-function test data
    if not args.skip_no_function:
        logger.info(f"\n=== Generating no-function test data ===")
        no_func_cases = generate_no_function(args.no_function_count)
        no_func_cases = add_test_ids(no_func_cases, "no_function")
        save_test_data(no_func_cases, "no_function.json")
        all_test_data.extend(no_func_cases)

    # Save combined dataset
    save_test_data(all_test_data, "all_test_data.json")

    # Summary
    print("\n" + "=" * 50)
    print("Test Data Generation Summary")
    print("=" * 50)
    for func_name in functions_to_generate:
        count = sum(1 for c in all_test_data if c.get("category") == func_name)
        print(f"  {func_name}: {count} cases")
    if not args.skip_no_function:
        no_func_count = sum(1 for c in all_test_data if c.get("category") == "no_function")
        print(f"  no_function: {no_func_count} cases")
    print(f"  TOTAL: {len(all_test_data)} cases")
    print(f"\nData saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()
