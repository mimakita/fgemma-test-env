"""PEFT LoRA Fine-tuned Model Evaluation Script.

Evaluates the PEFT fine-tuned FunctionGemma model on test data.

Usage:
    python -m tools.evaluate_peft --run-id 1
    python -m tools.evaluate_peft --run-id 1 --checkpoint 100
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

    # Add tools
    tools_text = "Available tools:\n"
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "")
        desc = func.get("description", "")
        tools_text += f"- {name}: {desc}\n"
    parts.append(tools_text)

    # Add conversation
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
    # Try to parse as JSON
    try:
        data = json.loads(response.strip())
        if isinstance(data, dict) and "name" in data:
            return data["name"]
    except:
        pass

    # Look for function names in response
    for func_name in function_names:
        if func_name in response:
            return func_name

    return "no_function"


def main():
    parser = argparse.ArgumentParser(description="Evaluate PEFT Fine-tuned Model")
    parser.add_argument("--run-id", type=int, required=True, help="Run ID")
    parser.add_argument("--checkpoint", type=int, help="Checkpoint step (optional)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base model")
    args = parser.parse_args()

    run_dir = FINETUNE_DIR / f"run_{args.run_id}"
    adapter_dir = PEFT_ADAPTERS_DIR / f"run_{args.run_id}"

    if args.checkpoint:
        adapter_dir = adapter_dir / f"checkpoint-{args.checkpoint}"

    # Check paths exist
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        return
    if not adapter_dir.exists():
        logger.error(f"Adapter directory not found: {adapter_dir}")
        return

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

    # Evaluate
    results_by_func = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "total": 0})
    correct = 0
    total = 0

    logger.info("Starting evaluation...")
    for i, sample in enumerate(test_data):
        messages = sample.get("conversation", sample.get("messages", []))
        expected = sample.get("expected_function", "no_function")

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
    print(f"PEFT Fine-tuned Model Evaluation Results (Run {args.run_id})")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    print("=" * 70)
    print(f"\nOverall Accuracy: {correct/total*100:.1f}% ({correct}/{total})")

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

    # Save results
    results_dir = PROJECT_ROOT / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "run_id": args.run_id,
        "checkpoint": args.checkpoint,
        "accuracy": correct / total * 100,
        "total_samples": total,
        "correct": correct,
        "per_function": {
            func: {
                "tp": stats["tp"],
                "fp": stats["fp"],
                "fn": stats["fn"],
                "total": stats["total"],
                "precision": stats["tp"] / (stats["tp"] + stats["fp"]) * 100 if (stats["tp"] + stats["fp"]) > 0 else 0,
                "recall": stats["tp"] / (stats["tp"] + stats["fn"]) * 100 if (stats["tp"] + stats["fn"]) > 0 else 0,
            }
            for func, stats in results_by_func.items()
        }
    }

    checkpoint_suffix = f"_ckpt{args.checkpoint}" if args.checkpoint else ""
    results_file = results_dir / f"peft_run{args.run_id}{checkpoint_suffix}_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
