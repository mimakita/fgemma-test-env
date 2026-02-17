"""Analyze misclassification errors from PEFT evaluation results.

Identifies patterns in prediction errors to guide data improvement.

Usage:
    python -m tools.analyze_errors --run-id 2 --checkpoint 500
    python -m tools.analyze_errors --run-id 2 --checkpoint 500 --function translation_assist
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

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
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

DEFAULT_MODEL = "google/functiongemma-270m-it"


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
    parser = argparse.ArgumentParser(description="Analyze misclassification errors")
    parser.add_argument("--run-id", type=int, required=True, help="Run ID")
    parser.add_argument("--checkpoint", type=int, help="Checkpoint step")
    parser.add_argument("--function", type=str, help="Focus on specific function")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base model")
    parser.add_argument("--limit", type=int, default=20, help="Max errors to show per category")
    args = parser.parse_args()

    run_dir = FINETUNE_DIR / f"run_{args.run_id}"
    adapter_dir = PEFT_ADAPTERS_DIR / f"run_{args.run_id}"

    if args.checkpoint:
        adapter_dir = adapter_dir / f"checkpoint-{args.checkpoint}"

    # Load test data
    test_file = run_dir / "all_test_data.json"
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    logger.info(f"Loaded {len(test_data)} test samples")

    # Load tools
    tools = load_tools_from_training_data(run_dir)
    function_names = [t["function"]["name"] for t in tools]
    logger.info(f"Functions: {function_names}")

    # Load model
    logger.info(f"Loading base model: {args.model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    logger.info(f"Loading PEFT adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    # Collect errors
    errors = defaultdict(list)  # (expected, predicted) -> list of samples
    confusion = defaultdict(lambda: defaultdict(int))

    logger.info("Analyzing predictions...")
    for i, sample in enumerate(test_data):
        messages = sample.get("conversation", sample.get("messages", []))
        expected = sample.get("expected_function") or "no_function"

        # Filter by function if specified
        if args.function and expected != args.function:
            # Also include cases predicted as this function
            pass  # Will filter after prediction

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

        # Update confusion matrix
        confusion[expected][predicted] += 1

        # Collect error cases
        if predicted != expected:
            if args.function:
                # Only collect errors related to the specified function
                if expected == args.function or predicted == args.function:
                    errors[(expected, predicted)].append({
                        "sample": sample,
                        "response": response[:200],
                        "messages": messages,
                    })
            else:
                errors[(expected, predicted)].append({
                    "sample": sample,
                    "response": response[:200],
                    "messages": messages,
                })

        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i+1}/{len(test_data)}")

    # Print confusion matrix
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)

    all_labels = sorted(set(list(confusion.keys()) + [k for v in confusion.values() for k in v.keys()]))

    # Header
    header = f"{'Expected':<20}"
    for label in all_labels:
        short_label = label[:8] if len(label) > 8 else label
        header += f"{short_label:>10}"
    print(header)
    print("-" * 80)

    # Rows
    for expected in all_labels:
        row = f"{expected:<20}"
        for predicted in all_labels:
            count = confusion[expected][predicted]
            if count > 0:
                if expected == predicted:
                    row += f"  [{count:>4}]  "
                else:
                    row += f"{count:>10}"
            else:
                row += f"{'':>10}"
        print(row)

    # Print error analysis
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS")
    print("=" * 80)

    # Sort errors by count
    sorted_errors = sorted(errors.items(), key=lambda x: -len(x[1]))

    for (expected, predicted), cases in sorted_errors:
        if args.function and expected != args.function and predicted != args.function:
            continue

        print(f"\n--- {expected} -> {predicted} ({len(cases)} cases) ---")

        for i, case in enumerate(cases[:args.limit]):
            messages = case["messages"]
            print(f"\n  [{i+1}] Conversation:")
            for msg in messages[-3:]:  # Last 3 messages
                role = msg.get("role", "?")
                content = msg.get("content", "")[:100]
                print(f"      {role}: {content}")
            print(f"      Model output: {case['response'][:100]}")

    # Summary
    print("\n" + "=" * 80)
    print("ERROR SUMMARY")
    print("=" * 80)

    total_errors = sum(len(cases) for cases in errors.values())
    total_samples = len(test_data)
    print(f"Total errors: {total_errors}/{total_samples} ({total_errors/total_samples*100:.1f}%)")

    print("\nTop error patterns:")
    for (expected, predicted), cases in sorted_errors[:10]:
        print(f"  {expected} -> {predicted}: {len(cases)} cases")

    # Save detailed results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_suffix = f"_ckpt{args.checkpoint}" if args.checkpoint else ""
    func_suffix = f"_{args.function}" if args.function else ""

    results_file = RESULTS_DIR / f"error_analysis_run{args.run_id}{checkpoint_suffix}{func_suffix}.json"

    results = {
        "run_id": args.run_id,
        "checkpoint": args.checkpoint,
        "focus_function": args.function,
        "total_samples": total_samples,
        "total_errors": total_errors,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "error_counts": {f"{e}->{p}": len(cases) for (e, p), cases in errors.items()},
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
