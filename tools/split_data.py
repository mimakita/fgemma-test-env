"""Split test data into train/test sets and convert to mlx-lm JSONL format.

Performs stratified 80:20 split preserving category proportions.
Converts training data to mlx-lm chat JSONL format with tool_calls.

Features:
- Balanced training data (undersamples no_function to match function count)
- Data augmentation for underrepresented functions
- Stratified split preserving category proportions

Usage:
    python -m tools.split_data --run-id 1 --seed 42
    python -m tools.split_data --run-id 2 --seed 123
    python -m tools.split_data --run-id 1 --seed 42 --balance  # Enable balancing
"""

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

from src.config import FINETUNE_TRAIN_RATIO
from src.functions import init_all, registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "test"
FINETUNE_DIR = PROJECT_ROOT / "data" / "finetune"

FUNCTION_NAMES = [
    "travel_guide",
    "celebrity_info",
    "shopping_intent",
    "sentiment_label",
    "weather_info",
    "schedule_reminder",
    "translation_assist",
]


def load_all_data() -> dict[str, list[dict]]:
    """Load all test data grouped by category."""
    data_by_category = {}

    for func_name in FUNCTION_NAMES:
        filepath = DATA_DIR / f"{func_name}.json"
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                data_by_category[func_name] = json.load(f)
            logger.info(f"Loaded {len(data_by_category[func_name])} cases for {func_name}")
        else:
            logger.warning(f"No data file for {func_name}")
            data_by_category[func_name] = []

    filepath = DATA_DIR / "no_function.json"
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            data_by_category["no_function"] = json.load(f)
        logger.info(f"Loaded {len(data_by_category['no_function'])} cases for no_function")
    else:
        data_by_category["no_function"] = []

    return data_by_category


from typing import Optional

def balance_training_data(
    train_data: list[dict],
    seed: int,
    target_per_function: Optional[int] = None,
    no_function_ratio: Optional[float] = None,
) -> list[dict]:
    """Balance training data by adjusting class proportions.

    Args:
        train_data: List of training samples
        seed: Random seed
        target_per_function: Target count per function (if None, uses median)
        no_function_ratio: Target ratio for no_function (e.g., 0.5 = 50%)
                          If set, overrides target_per_function for no_function

    Returns:
        Balanced training data
    """
    rng = random.Random(seed)

    # Group by category
    by_category = {}
    for sample in train_data:
        msgs = sample.get("messages", [])
        last_msg = msgs[-1] if msgs else {}
        if last_msg.get("tool_calls"):
            cat = last_msg["tool_calls"][0]["function"]["name"]
        else:
            cat = "no_function"

        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(sample)

    # Calculate target count for functions
    function_counts = [len(v) for k, v in by_category.items() if k != "no_function"]
    if target_per_function is None:
        target_per_function = int(sum(function_counts) / len(function_counts))

    # Calculate no_function target based on ratio if specified
    if no_function_ratio is not None:
        # no_function_ratio = no_function_count / total_count
        # no_function_count = no_function_ratio * total_count
        # total_count = function_count + no_function_count
        # function_count = num_functions * target_per_function
        num_functions = len([k for k in by_category.keys() if k != "no_function"])
        total_function_count = num_functions * target_per_function
        # ratio = no_function / (no_function + function)
        # ratio * (no_function + function) = no_function
        # ratio * function = no_function * (1 - ratio)
        # no_function = ratio * function / (1 - ratio)
        no_function_target = int(no_function_ratio * total_function_count / (1 - no_function_ratio))
        logger.info(f"No-function ratio: {no_function_ratio:.1%} -> target {no_function_target} no_function samples")
    else:
        no_function_target = target_per_function

    logger.info(f"Balancing: target {target_per_function} per function, {no_function_target} for no_function")

    balanced = []
    for cat, samples in by_category.items():
        target = no_function_target if cat == "no_function" else target_per_function

        if len(samples) > target:
            # Undersample
            sampled = rng.sample(samples, target)
            logger.info(f"  {cat}: {len(samples)} -> {len(sampled)} (undersampled)")
        elif len(samples) < target:
            # Oversample (with replacement)
            sampled = samples.copy()
            while len(sampled) < target:
                sampled.append(rng.choice(samples))
            logger.info(f"  {cat}: {len(samples)} -> {len(sampled)} (oversampled)")
        else:
            sampled = samples
            logger.info(f"  {cat}: {len(samples)} (unchanged)")

        balanced.extend(sampled)

    rng.shuffle(balanced)

    # Log final ratio
    final_no_function = sum(1 for s in balanced if not s.get("messages", [{}])[-1].get("tool_calls"))
    final_ratio = final_no_function / len(balanced) if balanced else 0
    logger.info(f"Final no_function ratio: {final_ratio:.1%} ({final_no_function}/{len(balanced)})")

    return balanced


def stratified_split(
    data_by_category: dict[str, list[dict]],
    train_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Split data maintaining category proportions."""
    rng = random.Random(seed)

    train_data = []
    test_data = []

    for category, cases in data_by_category.items():
        shuffled = list(cases)
        rng.shuffle(shuffled)

        split_idx = int(len(shuffled) * train_ratio)
        train_data.extend(shuffled[:split_idx])
        test_data.extend(shuffled[split_idx:])

        logger.info(
            f"  {category}: {len(shuffled)} total -> "
            f"{split_idx} train + {len(shuffled) - split_idx} test"
        )

    # Shuffle final sets
    rng.shuffle(train_data)
    rng.shuffle(test_data)

    return train_data, test_data


def get_tool_schemas() -> list[dict]:
    """Get all function tool schemas in OpenAI-compatible format."""
    init_all()
    tools = []
    for func_def in registry.get_all_definitions():
        tools.append({
            "type": "function",
            "function": {
                "name": func_def.name,
                "description": func_def.description,
                "parameters": func_def.parameters,
            },
        })
    return tools


def case_to_chat_jsonl(case: dict, tools: list[dict]) -> dict:
    """Convert a test case to mlx-lm chat JSONL format.

    Format for function calls:
    {"messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "...", "arguments": {...}}}]}
    ], "tools": [...]}

    Format for no_function:
    {"messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "I'll help you with that..."}
    ], "tools": [...]}
    """
    conversation = case.get("conversation", [])
    expected_function = case.get("expected_function")
    expected_args = case.get("expected_arguments") or {}

    messages = list(conversation)

    if expected_function:
        # Add assistant tool_call response
        tool_call = {
            "type": "function",
            "function": {
                "name": expected_function,
                "arguments": json.dumps(expected_args, ensure_ascii=False),
            },
        }
        messages.append({
            "role": "assistant",
            "tool_calls": [tool_call],
        })
    else:
        # No function - add a simple text response
        messages.append({
            "role": "assistant",
            "content": "はい、お手伝いします。",
        })

    return {
        "messages": messages,
        "tools": tools,
    }


def save_jsonl(data: list[dict], filepath: Path):
    """Save data as JSONL."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(data)} entries to {filepath}")


def print_data_distribution(data: list[dict], label: str):
    """Print distribution of categories in data."""
    categories = []
    for sample in data:
        msgs = sample.get("messages", [])
        last_msg = msgs[-1] if msgs else {}
        if last_msg.get("tool_calls"):
            cat = last_msg["tool_calls"][0]["function"]["name"]
        else:
            cat = "no_function"
        categories.append(cat)

    print(f"\n{label} distribution:")
    for cat, count in sorted(Counter(categories).items()):
        print(f"  {cat}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Split data and convert to JSONL")
    parser.add_argument("--run-id", type=int, required=True, help="Run ID (1 or 2)")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for split")
    parser.add_argument("--balance", action="store_true", help="Balance training data")
    parser.add_argument("--target-count", type=int, help="Target count per category for balancing")
    parser.add_argument(
        "--no-function-ratio", type=float, default=None,
        help="Target ratio for no_function (e.g., 0.5 = 50%%). Requires --balance"
    )
    args = parser.parse_args()

    run_dir = FINETUNE_DIR / f"run_{args.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading test data...")
    data_by_category = load_all_data()

    total = sum(len(v) for v in data_by_category.values())
    logger.info(f"Total cases: {total}")

    # Stratified split
    logger.info(f"\nSplitting with ratio {FINETUNE_TRAIN_RATIO} (seed={args.seed})...")
    train_data, test_data = stratified_split(
        data_by_category, FINETUNE_TRAIN_RATIO, args.seed
    )

    logger.info(f"\nTrain: {len(train_data)}, Test: {len(test_data)}")

    # Save test data in evaluate.py compatible format
    test_file = run_dir / "all_test_data.json"
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved test data: {test_file}")

    # Convert train data to JSONL
    tools = get_tool_schemas()
    logger.info(f"Using {len(tools)} tool schemas")

    train_jsonl = [case_to_chat_jsonl(case, tools) for case in train_data]

    # Balance training data if requested
    if args.balance:
        logger.info("\nBalancing training data...")
        train_jsonl = balance_training_data(
            train_jsonl, args.seed, args.target_count, args.no_function_ratio
        )

    # Print distribution before/after balancing
    print_data_distribution(train_jsonl, "Training data")

    # Split train into train/valid (90:10)
    rng = random.Random(args.seed + 1000)
    rng.shuffle(train_jsonl)
    valid_size = max(1, int(len(train_jsonl) * 0.1))
    valid_jsonl = train_jsonl[:valid_size]
    train_jsonl = train_jsonl[valid_size:]

    # Save
    save_jsonl(train_jsonl, run_dir / "train.jsonl")
    save_jsonl(valid_jsonl, run_dir / "valid.jsonl")

    # Summary
    print("\n" + "=" * 60)
    print(f"Data Split Summary (Run {args.run_id}, seed={args.seed})")
    print("=" * 60)
    print(f"  Total cases:     {total}")
    print(f"  Train (JSONL):   {len(train_jsonl)}")
    print(f"  Valid (JSONL):   {len(valid_jsonl)}")
    print(f"  Test (JSON):     {len(test_data)}")
    print(f"  Balanced:        {args.balance}")
    if args.no_function_ratio:
        print(f"  No-function ratio: {args.no_function_ratio:.1%}")
    print(f"\n  Output directory: {run_dir}")
    print(f"  Files:")
    print(f"    {run_dir / 'train.jsonl'}")
    print(f"    {run_dir / 'valid.jsonl'}")
    print(f"    {run_dir / 'all_test_data.json'}")

    # Save split metadata
    metadata = {
        "run_id": args.run_id,
        "seed": args.seed,
        "train_ratio": FINETUNE_TRAIN_RATIO,
        "total_cases": total,
        "train_count": len(train_jsonl),
        "valid_count": len(valid_jsonl),
        "test_count": len(test_data),
        "tools_count": len(tools),
        "balanced": args.balance,
        "target_count": args.target_count,
        "no_function_ratio": args.no_function_ratio,
    }
    with open(run_dir / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
