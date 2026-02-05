"""Filter test data to keep only Japanese conversations.

Usage:
    python -m tools.filter_japanese_only
"""

import json
import logging
import re
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "test"


def is_japanese_text(text: str) -> bool:
    """Check if text contains Japanese characters (hiragana, katakana, kanji)."""
    # Japanese character ranges
    japanese_pattern = re.compile(
        r'[\u3040-\u309F]|'  # Hiragana
        r'[\u30A0-\u30FF]|'  # Katakana
        r'[\u4E00-\u9FFF]'   # Kanji (CJK Unified Ideographs)
    )
    return bool(japanese_pattern.search(text))


def is_mostly_english(text: str) -> bool:
    """Check if text is mostly English (contains no Japanese characters)."""
    return not is_japanese_text(text)


def is_japanese_conversation(conversation: list[dict]) -> bool:
    """Check if the conversation is primarily in Japanese.

    A conversation is considered Japanese if ALL user messages contain Japanese.
    """
    user_messages = [m.get("content", "") for m in conversation if m.get("role") == "user"]

    if not user_messages:
        return False

    # All user messages must contain Japanese
    return all(is_japanese_text(msg) for msg in user_messages)


def filter_file(filepath: Path) -> tuple[int, int]:
    """Filter a single test data file. Returns (original, filtered)."""
    with open(filepath, "r", encoding="utf-8") as f:
        cases = json.load(f)

    original_count = len(cases)
    label = filepath.stem

    # Filter to Japanese only
    japanese_cases = [
        case for case in cases
        if is_japanese_conversation(case.get("conversation", []))
    ]

    # Re-number test IDs
    for i, case in enumerate(japanese_cases):
        case["test_id"] = f"{label}_{i + 1:04d}"

    filtered_count = len(japanese_cases)
    removed = original_count - filtered_count

    if removed > 0:
        logger.info(f"{label}: {original_count} → {filtered_count} (removed {removed} English cases)")
    else:
        logger.info(f"{label}: {filtered_count} (all Japanese)")

    # Save filtered data
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(japanese_cases, f, ensure_ascii=False, indent=2)

    return original_count, filtered_count


def main():
    # Function names
    function_names = [
        "travel_guide",
        "celebrity_info",
        "shopping_intent",
        "sentiment_label",
        "weather_info",
        "schedule_reminder",
        "translation_assist",
        "no_function",
    ]

    total_original = 0
    total_filtered = 0
    needs_regeneration = {}

    print("\n" + "=" * 60)
    print("Filtering to Japanese-only conversations")
    print("=" * 60 + "\n")

    all_filtered_cases = []

    for func_name in function_names:
        filepath = DATA_DIR / f"{func_name}.json"
        if filepath.exists():
            orig, filtered = filter_file(filepath)
            total_original += orig
            total_filtered += filtered

            # Check if regeneration needed
            target = 250 if func_name == "no_function" else 100
            if filtered < target:
                needs_regeneration[func_name] = target - filtered

            # Collect for combined file
            with open(filepath, "r", encoding="utf-8") as f:
                all_filtered_cases.extend(json.load(f))

    # Save combined file
    combined_path = DATA_DIR / "all_test_data.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_filtered_cases, f, ensure_ascii=False, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("Filter Summary")
    print("=" * 60)
    print(f"  Original cases: {total_original}")
    print(f"  Filtered cases: {total_filtered}")
    print(f"  Removed:        {total_original - total_filtered}")

    if needs_regeneration:
        print("\n  Functions needing regeneration:")
        for func_name, needed in needs_regeneration.items():
            print(f"    - {func_name}: needs {needed} more cases")
    else:
        print("\n  All functions have sufficient data!")


if __name__ == "__main__":
    main()
