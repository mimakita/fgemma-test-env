"""Fix test data quality issues.

Priority 1: Ensure all conversations end with a user message
Priority 2: Limit turn count to 2-4, remove function name leaks

Usage:
    python -m tools.fix_test_data
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "test"
BACKUP_DIR = Path(__file__).parent.parent / "data" / "test_backup"

# Function names to detect leaks
FUNCTION_NAMES = [
    "travel_guide",
    "celebrity_info",
    "shopping_intent",
    "sentiment_label",
    "weather_info",
    "schedule_reminder",
    "translation_assist",
]


def has_function_name_leak(conversation: list[dict]) -> bool:
    """Check if conversation contains function name leaks."""
    for msg in conversation:
        content = msg.get("content", "").lower()
        for func_name in FUNCTION_NAMES:
            # Check for function name mentions (with some tolerance for natural mentions)
            if func_name in content:
                # Check if it looks like a technical leak
                if any(pattern in content for pattern in [
                    f"{func_name}(",
                    f"{func_name} パラメータ",
                    f"{func_name} 関数",
                    f"「{func_name}」",
                    f'"{func_name}"',
                    f"'{func_name}'",
                    f"{func_name}を使",
                    f"{func_name}を呼",
                ]):
                    return True
    return False


def fix_conversation_ending(conversation: list[dict]) -> list[dict]:
    """Ensure conversation ends with a user message.

    Strategy:
    - If already ends with user, keep as is
    - If ends with assistant, remove trailing assistant messages until we hit a user message
    - If that leaves us with < 2 messages or no user messages, mark for regeneration
    """
    if not conversation:
        return []

    # Already ends with user
    if conversation[-1].get("role") == "user":
        return conversation

    # Remove trailing assistant messages
    fixed = list(conversation)
    while fixed and fixed[-1].get("role") == "assistant":
        fixed.pop()

    # Validate: need at least 2 messages with at least 1 user
    if len(fixed) < 2:
        return []
    if not any(m.get("role") == "user" for m in fixed):
        return []

    return fixed


def fix_turn_count(conversation: list[dict], min_turns: int = 2, max_turns: int = 4) -> list[dict]:
    """Limit conversation to 2-4 turns (messages).

    If too long, truncate from the beginning while keeping:
    - At least min_turns messages
    - At most max_turns messages
    - Must end with user message
    """
    if not conversation:
        return []

    # If within limits, return as is
    if min_turns <= len(conversation) <= max_turns:
        return conversation

    # If too short, mark for regeneration
    if len(conversation) < min_turns:
        return []

    # Too long - truncate from beginning
    # Find a valid starting point that gives us max_turns messages ending with user
    for start_idx in range(len(conversation) - max_turns, len(conversation) - min_turns + 1):
        if start_idx < 0:
            continue
        truncated = conversation[start_idx:start_idx + max_turns]
        # Ensure it ends with user
        if truncated and truncated[-1].get("role") == "user":
            return truncated

    # Fallback: take last max_turns and fix ending
    truncated = conversation[-max_turns:]
    return fix_conversation_ending(truncated)


def fix_test_case(case: dict) -> Optional[dict]:
    """Fix a single test case. Returns None if unfixable."""
    conversation = case.get("conversation", [])

    # Check for function name leaks
    if has_function_name_leak(conversation):
        logger.debug(f"Removing case with function name leak: {case.get('test_id')}")
        return None

    # Fix ending (must end with user)
    fixed_conv = fix_conversation_ending(conversation)
    if not fixed_conv:
        logger.debug(f"Removing case with unfixable ending: {case.get('test_id')}")
        return None

    # Fix turn count (2-4 turns)
    fixed_conv = fix_turn_count(fixed_conv)
    if not fixed_conv:
        logger.debug(f"Removing case with invalid turn count: {case.get('test_id')}")
        return None

    # Return fixed case
    fixed_case = dict(case)
    fixed_case["conversation"] = fixed_conv
    return fixed_case


def analyze_data(cases: list[dict], label: str) -> dict:
    """Analyze test data characteristics."""
    total = len(cases)
    if total == 0:
        return {"total": 0}

    last_user = sum(1 for c in cases if c.get("conversation", [])[-1].get("role") == "user")
    last_asst = total - last_user

    turn_counts = [len(c.get("conversation", [])) for c in cases]
    avg_turns = sum(turn_counts) / total if total > 0 else 0

    leaks = sum(1 for c in cases if has_function_name_leak(c.get("conversation", [])))

    return {
        "total": total,
        "last_user": last_user,
        "last_assistant": last_asst,
        "avg_turns": round(avg_turns, 1),
        "min_turns": min(turn_counts) if turn_counts else 0,
        "max_turns": max(turn_counts) if turn_counts else 0,
        "function_name_leaks": leaks,
    }


def process_file(filepath: Path) -> tuple[int, int, int]:
    """Process a single test data file. Returns (original, fixed, removed)."""
    with open(filepath, "r", encoding="utf-8") as f:
        cases = json.load(f)

    original_count = len(cases)
    label = filepath.stem

    # Analyze before
    before_stats = analyze_data(cases, label)
    logger.info(f"\n=== {label} ===")
    logger.info(f"Before: {before_stats}")

    # Fix cases
    fixed_cases = []
    for case in cases:
        fixed = fix_test_case(case)
        if fixed:
            fixed_cases.append(fixed)

    # Re-number test IDs
    for i, case in enumerate(fixed_cases):
        case["test_id"] = f"{label}_{i + 1:04d}"

    # Analyze after
    after_stats = analyze_data(fixed_cases, label)
    logger.info(f"After:  {after_stats}")

    removed = original_count - len(fixed_cases)
    if removed > 0:
        logger.warning(f"Removed {removed} cases from {label}")

    # Save fixed data
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(fixed_cases, f, ensure_ascii=False, indent=2)

    return original_count, len(fixed_cases), removed


def main():
    # Create backup
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Creating backup in {BACKUP_DIR}")

    for filepath in DATA_DIR.glob("*.json"):
        backup_path = BACKUP_DIR / filepath.name
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # Process individual function files
    total_original = 0
    total_fixed = 0
    total_removed = 0

    all_fixed_cases = []

    for func_name in FUNCTION_NAMES + ["no_function"]:
        filepath = DATA_DIR / f"{func_name}.json"
        if filepath.exists():
            orig, fixed, removed = process_file(filepath)
            total_original += orig
            total_fixed += fixed
            total_removed += removed

            # Collect for combined file
            with open(filepath, "r", encoding="utf-8") as f:
                all_fixed_cases.extend(json.load(f))

    # Save combined file
    combined_path = DATA_DIR / "all_test_data.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_fixed_cases, f, ensure_ascii=False, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("Test Data Fix Summary")
    print("=" * 60)
    print(f"  Original cases: {total_original}")
    print(f"  Fixed cases:    {total_fixed}")
    print(f"  Removed cases:  {total_removed} ({total_removed/total_original*100:.1f}%)")
    print(f"\nBackup saved to: {BACKUP_DIR}")
    print(f"Fixed data saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()
