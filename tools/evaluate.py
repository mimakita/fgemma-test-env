"""Evaluation script - measures FunctionGemma routing accuracy against test data.

Usage:
    python -m tools.evaluate [--data-dir PATH] [--output-dir PATH]
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import ROUTER_MODEL
from src.ollama_client import OllamaClient
from src.functions import init_all, registry
from src.router import FunctionRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "test"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "results"


def load_test_data(data_dir: Path) -> list[dict]:
    """Load all test data from JSON files."""
    all_data = []

    # Prefer all_test_data.json if it exists
    combined = data_dir / "all_test_data.json"
    if combined.exists():
        with open(combined, "r", encoding="utf-8") as f:
            all_data = json.load(f)
        logger.info(f"Loaded {len(all_data)} cases from {combined}")
        return all_data

    # Otherwise load individual files
    for json_file in sorted(data_dir.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
                logger.info(f"Loaded {len(data)} cases from {json_file.name}")

    logger.info(f"Total loaded: {len(all_data)} cases")
    return all_data


def evaluate_single(
    router: FunctionRouter,
    case: dict,
) -> dict:
    """Evaluate a single test case."""
    conversation = case.get("conversation", [])
    expected_function = case.get("expected_function")
    expected_args = case.get("expected_arguments")

    try:
        result = router.route(conversation)
    except Exception as e:
        logger.error(f"Error routing {case.get('test_id', '?')}: {e}")
        return {
            "test_id": case.get("test_id", "unknown"),
            "category": case.get("category", "unknown"),
            "expected_function": expected_function,
            "predicted_function": None,
            "expected_arguments": expected_args,
            "predicted_arguments": None,
            "function_correct": False,
            "args_correct": False,
            "error": str(e),
        }

    predicted_function = result.function_name if result.should_call else None
    predicted_args = result.arguments if result.should_call else None

    # Check function correctness
    function_correct = predicted_function == expected_function

    # Check argument correctness (partial match)
    args_correct = False
    if function_correct and expected_args and predicted_args:
        # Check if all expected required args are present and match
        args_correct = all(
            predicted_args.get(k) == v
            for k, v in expected_args.items()
            if k in (case.get("expected_arguments") or {})
        )
    elif function_correct and expected_function is None:
        args_correct = True  # No function, no args to check

    return {
        "test_id": case.get("test_id", "unknown"),
        "category": case.get("category", "unknown"),
        "expected_function": expected_function,
        "predicted_function": predicted_function,
        "expected_arguments": expected_args,
        "predicted_arguments": predicted_args,
        "function_correct": function_correct,
        "args_correct": args_correct,
        "raw_response": result.raw_response[:200] if result.raw_response else "",
    }


def compute_metrics(results: list[dict]) -> dict:
    """Compute precision, recall, F1 per function and overall metrics."""
    # Get all function names that appear in expected or predicted
    all_functions = set()
    for r in results:
        if r["expected_function"]:
            all_functions.add(r["expected_function"])
        if r["predicted_function"]:
            all_functions.add(r["predicted_function"])

    per_function = {}
    for func_name in sorted(all_functions):
        tp = sum(
            1 for r in results
            if r["expected_function"] == func_name and r["predicted_function"] == func_name
        )
        fp = sum(
            1 for r in results
            if r["expected_function"] != func_name and r["predicted_function"] == func_name
        )
        fn = sum(
            1 for r in results
            if r["expected_function"] == func_name and r["predicted_function"] != func_name
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_function[func_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "total_expected": tp + fn,
        }

    # Overall accuracy (function matching)
    total = len(results)
    correct = sum(1 for r in results if r["function_correct"])
    overall_accuracy = correct / total if total > 0 else 0.0

    # False positive rate (no function expected but function predicted)
    no_func_cases = [r for r in results if r["expected_function"] is None]
    false_positives = sum(1 for r in no_func_cases if r["predicted_function"] is not None)
    fpr = false_positives / len(no_func_cases) if no_func_cases else 0.0

    # Argument accuracy (among correctly routed cases)
    correct_func_cases = [r for r in results if r["function_correct"] and r["expected_function"]]
    args_correct = sum(1 for r in correct_func_cases if r["args_correct"])
    args_accuracy = args_correct / len(correct_func_cases) if correct_func_cases else 0.0

    return {
        "per_function": per_function,
        "overall": {
            "accuracy": round(overall_accuracy, 4),
            "false_positive_rate": round(fpr, 4),
            "argument_accuracy": round(args_accuracy, 4),
            "total_cases": total,
            "correct_cases": correct,
            "no_function_cases": len(no_func_cases),
            "false_positives": false_positives,
        },
    }


def print_report(metrics: dict):
    """Print a formatted evaluation report to console."""
    overall = metrics["overall"]
    per_func = metrics["per_function"]

    print("\n" + "=" * 70)
    print("FuncGemma Evaluation Report")
    print(f"Model: {ROUTER_MODEL}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print(f"\nOverall Metrics:")
    print(f"  Total test cases:      {overall['total_cases']}")
    print(f"  Function accuracy:     {overall['accuracy']:.1%}")
    print(f"  False positive rate:   {overall['false_positive_rate']:.1%}")
    print(f"  Argument accuracy:     {overall['argument_accuracy']:.1%}")
    print(f"  No-function cases:     {overall['no_function_cases']}")
    print(f"  False positives:       {overall['false_positives']}")

    print(f"\nPer-Function Results:")
    print("-" * 70)
    print(f"{'Function':<22} {'Prec':>6} {'Recall':>7} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'Total':>6}")
    print("-" * 70)

    for func_name, m in sorted(per_func.items()):
        print(
            f"{func_name:<22} {m['precision']:>6.2%} {m['recall']:>7.2%} "
            f"{m['f1']:>6.2%} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5} {m['total_expected']:>6}"
        )

    print("-" * 70)
    print()


def save_results(results: list[dict], metrics: dict, output_dir: Path):
    """Save detailed results and metrics to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results
    results_file = output_dir / f"results_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Detailed results saved to {results_file}")

    # Save metrics
    metrics_file = output_dir / f"metrics_{timestamp}.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")

    # Save latest symlink-like file
    latest_file = output_dir / "latest_metrics.json"
    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate FunctionGemma routing accuracy")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=f"Directory containing test data (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for evaluation results (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Initialize
    init_all()
    client = OllamaClient()
    router = FunctionRouter(client, registry)

    # Load test data
    test_data = load_test_data(data_dir)
    if not test_data:
        logger.error("No test data found!")
        sys.exit(1)

    # Run evaluation
    logger.info(f"Evaluating {len(test_data)} test cases...")
    results = []
    start_time = time.time()

    for i, case in enumerate(test_data):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(test_data) - i - 1) / rate if rate > 0 else 0
            logger.info(
                f"Progress: {i + 1}/{len(test_data)} "
                f"({(i + 1) / len(test_data):.0%}) "
                f"ETA: {eta:.0f}s"
            )

        result = evaluate_single(router, case)
        results.append(result)

    elapsed = time.time() - start_time
    logger.info(f"Evaluation completed in {elapsed:.1f}s")

    # Compute metrics
    metrics = compute_metrics(results)
    metrics["metadata"] = {
        "model": ROUTER_MODEL,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "data_dir": str(data_dir),
    }

    # Output
    print_report(metrics)
    save_results(results, metrics, output_dir)


if __name__ == "__main__":
    main()
