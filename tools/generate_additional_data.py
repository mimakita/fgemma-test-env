"""Generate additional test data to reach target counts.

Requirements:
- All Japanese
- Conversations must end with user message
- 2-4 turns only
- No function name leaks

Usage:
    python -m tools.generate_additional_data
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import ollama

from src.config import TEST_DATA_MODEL, TEST_DATA_OPTIONS
from src.functions import init_all, registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "test"

# Target counts (10% increase from 200/500)
TARGET_PER_FUNCTION = 220
TARGET_NO_FUNCTION = 550

# Function names
FUNCTION_NAMES = [
    "travel_guide",
    "celebrity_info",
    "shopping_intent",
    "sentiment_label",
    "weather_info",
    "schedule_reminder",
    "translation_assist",
]


def generate_function_prompt(func_name: str, description: str, parameters: dict, batch_size: int) -> str:
    """Create a prompt for generating test conversations for a specific function."""
    params_str = json.dumps(parameters, indent=2, ensure_ascii=False)
    return f"""日本語のテストデータを{batch_size}件生成。JSONオブジェクトで出力。"items"キーに配列を入れる。

関数: {func_name} - {description}
パラメータ: {params_str}

出力形式(このJSONだけ出力):
{{"items": [
  {{"conversation": [{{"role":"user","content":"ユーザーの日本語発言"}},{{"role":"assistant","content":"AIの日本語応答"}},{{"role":"user","content":"ユーザーの日本語発言2"}}], "expected_function":"{func_name}", "expected_arguments":{{"パラメータ名":"値"}}}},
  {{"conversation": [{{"role":"user","content":"別の日本語発言"}},{{"role":"assistant","content":"別の日本語応答"}},{{"role":"user","content":"ユーザーの日本語発言3"}}], "expected_function":"{func_name}", "expected_arguments":{{"パラメータ名":"値"}}}}
]}}

【厳守事項】
1. 全て日本語で生成（英語禁止）
2. 各対話は必ず「user」メッセージで終わる（最後は必ずユーザーの発言）
3. ターン数は2〜4（最小: user→assistant→user、最大: user→assistant→user→assistant→user→user）
4. 関数名「{func_name}」を対話内容に含めない
5. 現実的で自然な日本語の会話
6. 必ず{batch_size}件生成"""


def generate_no_function_prompt(function_names: list[str], batch_size: int) -> str:
    """Create a prompt for generating conversations that should NOT trigger any function."""
    funcs_str = ", ".join(function_names)
    return f"""以下の関数のどれにも該当しない通常の日本語対話を{batch_size}件生成。JSONオブジェクトで出力。"items"キーに配列を入れる。

該当しない関数: {funcs_str}

出力形式(このJSONだけ出力):
{{"items": [
  {{"conversation": [{{"role":"user","content":"ユーザーの日本語発言"}},{{"role":"assistant","content":"AIの日本語応答"}},{{"role":"user","content":"ユーザーの日本語発言2"}}], "expected_function":null, "expected_arguments":null}},
  {{"conversation": [{{"role":"user","content":"別の日本語発言"}},{{"role":"assistant","content":"別の日本語応答"}},{{"role":"user","content":"ユーザーの日本語発言3"}}], "expected_function":null, "expected_arguments":null}}
]}}

【厳守事項】
1. 全て日本語で生成（英語禁止）
2. 各対話は必ず「user」メッセージで終わる（最後は必ずユーザーの発言）
3. ターン数は2〜4
4. カテゴリ(挨拶/雑談/一般知識/意見交換/数学/創作/個人相談/技術質問/日常)から多様に
5. どの関数にも非該当
6. 必ず{batch_size}件生成"""


def extract_json_from_response(text: str) -> Optional[list]:
    """Extract JSON array from LLM response."""
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("items", "data", "test_data", "cases", "results", "conversations"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            if "conversation" in data:
                return [data]
        return None
    except json.JSONDecodeError:
        pass

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


def is_japanese_text(text: str) -> bool:
    """Check if text contains Japanese characters."""
    japanese_pattern = re.compile(
        r'[\u3040-\u309F]|[\u30A0-\u30FF]|[\u4E00-\u9FFF]'
    )
    return bool(japanese_pattern.search(text))


def has_function_name_leak(conversation: list[dict], func_name: Optional[str]) -> bool:
    """Check if conversation contains function name leaks."""
    if func_name is None:
        return False
    for msg in conversation:
        content = msg.get("content", "").lower()
        if func_name.lower() in content:
            return True
    return False


def validate_test_case(case: dict, func_name: Optional[str]) -> bool:
    """Validate a single test case."""
    if not isinstance(case, dict):
        return False

    conv = case.get("conversation")
    if not isinstance(conv, list) or len(conv) < 2 or len(conv) > 4:
        return False

    # Check each message
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

    # Must end with user message
    if conv[-1].get("role") != "user":
        return False

    # All user messages must be Japanese
    user_messages = [m["content"] for m in conv if m["role"] == "user"]
    if not all(is_japanese_text(msg) for msg in user_messages):
        return False

    # No function name leaks
    if has_function_name_leak(conv, func_name):
        return False

    # Validate expected function
    expected = case.get("expected_function")
    if func_name is not None:
        if expected != func_name:
            return False
    else:
        if expected is not None and expected not in (None, "null", "none", ""):
            return False

    return True


def generate_batch(
    prompt: str,
    func_name: Optional[str],
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
                format="json",
            )
            text = response.message.content or ""
            data = extract_json_from_response(text)

            if data is None:
                logger.warning(f"  Batch {batch_num} attempt {attempt + 1}: Failed to parse JSON")
                continue

            valid = []
            for case in data:
                if validate_test_case(case, func_name):
                    if func_name is None:
                        case["expected_function"] = None
                        case["expected_arguments"] = None
                    valid.append(case)

            if valid:
                logger.info(f"  Batch {batch_num}: {len(valid)}/{len(data)} valid cases")
                return valid
            else:
                logger.warning(f"  Batch {batch_num} attempt {attempt + 1}: No valid cases from {len(data)}")

        except Exception as e:
            logger.error(f"  Batch {batch_num} attempt {attempt + 1} error: {e}")
            time.sleep(2)

    logger.error(f"  Batch {batch_num}: Failed after {max_retries} retries")
    return []


def generate_for_function(func_name: str, target_count: int, existing_count: int) -> list[dict]:
    """Generate additional test data for a specific function."""
    func_def = registry.get(func_name)
    if not func_def:
        logger.error(f"Unknown function: {func_name}")
        return []

    needed = target_count - existing_count
    if needed <= 0:
        return []

    logger.info(f"\n=== Generating {needed} additional cases for: {func_name} ===")

    all_cases = []
    batch_num = 0
    batch_size = 5  # Smaller batches for better quality

    while len(all_cases) < needed:
        remaining = needed - len(all_cases)
        current_batch_size = min(batch_size, remaining)

        prompt = generate_function_prompt(
            func_name, func_def.description, func_def.parameters, current_batch_size
        )

        batch_num += 1
        cases = generate_batch(prompt, func_name, batch_num)
        all_cases.extend(cases)

        logger.info(f"  Total for {func_name}: {len(all_cases)}/{needed}")

        # Safety limit
        if batch_num > 50:
            logger.warning(f"  Reached batch limit for {func_name}")
            break

    return all_cases[:needed]


def generate_no_function(target_count: int, existing_count: int) -> list[dict]:
    """Generate additional no-function test data."""
    needed = target_count - existing_count
    if needed <= 0:
        return []

    logger.info(f"\n=== Generating {needed} additional no_function cases ===")

    all_func_names = registry.get_all_names()
    all_cases = []
    batch_num = 0
    batch_size = 5

    while len(all_cases) < needed:
        remaining = needed - len(all_cases)
        current_batch_size = min(batch_size, remaining)

        prompt = generate_no_function_prompt(all_func_names, current_batch_size)

        batch_num += 1
        cases = generate_batch(prompt, None, batch_num)
        all_cases.extend(cases)

        logger.info(f"  Total for no_function: {len(all_cases)}/{needed}")

        if batch_num > 80:
            logger.warning("  Reached batch limit for no_function")
            break

    return all_cases[:needed]


def add_test_ids(cases: list[dict], prefix: str, start_idx: int) -> list[dict]:
    """Add unique test IDs to each case."""
    for i, case in enumerate(cases):
        case["test_id"] = f"{prefix}_{start_idx + i + 1:04d}"
        case["category"] = prefix
    return cases


def main():
    # Initialize function registry
    init_all()

    # Check current counts
    print("\n" + "=" * 60)
    print("Current Data Status")
    print("=" * 60)

    needs_generation = {}

    for func_name in FUNCTION_NAMES:
        filepath = DATA_DIR / f"{func_name}.json"
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                current = len(json.load(f))
        else:
            current = 0

        target = TARGET_PER_FUNCTION
        if current < target:
            needs_generation[func_name] = (current, target)
            print(f"  {func_name}: {current}/{target} (needs {target - current})")
        else:
            print(f"  {func_name}: {current}/{target} ✓")

    # Check no_function
    filepath = DATA_DIR / "no_function.json"
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            current = len(json.load(f))
    else:
        current = 0

    target = TARGET_NO_FUNCTION
    if current < target:
        needs_generation["no_function"] = (current, target)
        print(f"  no_function: {current}/{target} (needs {target - current})")
    else:
        print(f"  no_function: {current}/{target} ✓")

    if not needs_generation:
        print("\nAll functions have sufficient data!")
        return

    # Generate additional data
    print("\n" + "=" * 60)
    print("Generating Additional Data")
    print("=" * 60)

    all_new_cases = []

    for func_name, (existing, target) in needs_generation.items():
        filepath = DATA_DIR / f"{func_name}.json"

        # Load existing cases
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                existing_cases = json.load(f)
        else:
            existing_cases = []

        # Generate new cases
        if func_name == "no_function":
            new_cases = generate_no_function(target, existing)
        else:
            new_cases = generate_for_function(func_name, target, existing)

        # Add IDs and combine
        new_cases = add_test_ids(new_cases, func_name, len(existing_cases))
        combined = existing_cases + new_cases

        # Save
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(combined)} total cases for {func_name}")
        all_new_cases.extend(new_cases)

    # Update combined file
    all_cases = []
    for func_name in FUNCTION_NAMES + ["no_function"]:
        filepath = DATA_DIR / f"{func_name}.json"
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                all_cases.extend(json.load(f))

    combined_path = DATA_DIR / "all_test_data.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("Generation Summary")
    print("=" * 60)
    print(f"  New cases generated: {len(all_new_cases)}")
    print(f"  Total cases now: {len(all_cases)}")

    print("\n  Final counts:")
    for func_name in FUNCTION_NAMES + ["no_function"]:
        filepath = DATA_DIR / f"{func_name}.json"
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                count = len(json.load(f))
            target = TARGET_NO_FUNCTION if func_name == "no_function" else TARGET_PER_FUNCTION
            status = "✓" if count >= target else f"({target - count} short)"
            print(f"    {func_name}: {count} {status}")


if __name__ == "__main__":
    main()
