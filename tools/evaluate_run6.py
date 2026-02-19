"""Run 6 evaluation: Two-stage pipeline (ML Stage 1 + PEFT Run5 Stage 2).

Stage 1: TF-IDF + LinearSVC (data/classifiers/stage1_model.pkl)
Stage 2: FunctionGemma Run5 checkpoint-800

Evaluates on the same test set as previous runs (run_5/all_test_data.json).

Usage:
    source .venv-ft/bin/activate
    python -m tools.evaluate_run6
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.classifier import FunctionCallClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
FINETUNE_DIR = PROJECT_ROOT / "data" / "finetune"
PEFT_ADAPTERS_DIR = PROJECT_ROOT / "data" / "peft_adapters"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

# Use Run5 Stage 2 model
STAGE2_RUN_ID = 5
STAGE2_CHECKPOINT = 800
BASE_MODEL = "google/functiongemma-270m-it"


def load_tools_from_training_data(run_dir: Path) -> list[dict]:
    train_file = run_dir / "train.jsonl"
    with open(train_file, "r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())
        return first_line.get("tools", [])


def format_prompt(messages: list[dict], tools: list[dict]) -> str:
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
    try:
        data = json.loads(response.strip())
        if isinstance(data, dict) and "name" in data:
            return data["name"]
    except Exception:
        pass

    for func_name in function_names:
        if func_name in response:
            return func_name

    return "no_function"


def main():
    run_dir = FINETUNE_DIR / f"run_{STAGE2_RUN_ID}"
    adapter_dir = PEFT_ADAPTERS_DIR / f"run_{STAGE2_RUN_ID}" / f"checkpoint-{STAGE2_CHECKPOINT}"

    # Load test data
    test_file = run_dir / "all_test_data.json"
    test_data = json.loads(test_file.read_text())
    logger.info(f"Test samples: {len(test_data)}")

    # Load tools / function names
    tools = load_tools_from_training_data(run_dir)
    function_names = [t["function"]["name"] for t in tools]
    logger.info(f"Functions: {function_names}")

    # Stage 1: ML classifier
    logger.info("Loading Stage 1 classifier (TF-IDF + LinearSVC)...")
    classifier = FunctionCallClassifier()
    logger.info(f"  Backend: {classifier.backend}")

    # Stage 2: PEFT model
    logger.info(f"Loading Stage 2 base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    logger.info(f"Loading PEFT adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    # Evaluate
    results_by_func = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "total": 0})
    correct = 0
    total = 0
    stage1_blocked = 0   # correctly blocked by Stage 1
    stage1_fp = 0        # Stage 1 false positive (should be no_function but passed to Stage 2)
    stage1_fn = 0        # Stage 1 false negative (needed function but blocked)
    stage2_called = 0

    latencies_s1 = []
    latencies_s2 = []

    logger.info("Starting Run 6 evaluation...")
    for i, sample in enumerate(test_data):
        messages = sample.get("conversation", sample.get("messages", []))
        expected = sample.get("expected_function", "no_function")

        # Normalize: expected_function=None means no_function
        if expected is None:
            expected = "no_function"

        results_by_func[expected]["total"] += 1

        # ── Stage 1 ──
        t0 = time.perf_counter()
        clf_result = classifier.classify(messages)
        latencies_s1.append(time.perf_counter() - t0)

        if not clf_result.need_function:
            # Stage 1 says: no function
            predicted = "no_function"
            stage1_blocked += 1
        else:
            # ── Stage 2 ──
            stage2_called += 1
            prompt = format_prompt(messages, tools)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            t1 = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            latencies_s2.append(time.perf_counter() - t1)

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            predicted = extract_function_from_response(response, function_names)

        # Stage 1 error tracking
        if expected == "no_function" and clf_result.need_function:
            stage1_fp += 1
        if expected != "no_function" and not clf_result.need_function:
            stage1_fn += 1

        # Accuracy
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

        if (i + 1) % 100 == 0:
            logger.info(
                f"Progress: {i+1}/{len(test_data)}, Accuracy: {correct/total*100:.1f}%"
                f"  (Stage2 calls: {stage2_called})"
            )

    # ── Print results ──
    avg_s1 = sum(latencies_s1) / len(latencies_s1) * 1000
    avg_s2 = sum(latencies_s2) / len(latencies_s2) * 1000 if latencies_s2 else 0

    print("\n" + "=" * 70)
    print("RUN 6: Two-Stage Evaluation")
    print(f"  Stage 1: TF-IDF + LinearSVC  ({classifier.backend})")
    print(f"  Stage 2: FunctionGemma PEFT Run{STAGE2_RUN_ID} ckpt-{STAGE2_CHECKPOINT}")
    print("=" * 70)
    print(f"\nOverall Accuracy : {correct/total*100:.1f}%  ({correct}/{total})")
    print(f"\nStage 1 stats:")
    print(f"  Blocked (no_function) : {stage1_blocked:4d} / {total}")
    print(f"  Passed to Stage 2     : {stage2_called:4d} / {total}")
    print(f"  Stage 1 FP (should block but didn't) : {stage1_fp}")
    print(f"  Stage 1 FN (blocked when func needed): {stage1_fn}")
    print(f"\nLatency (avg):")
    print(f"  Stage 1 : {avg_s1:.3f} ms/sample")
    print(f"  Stage 2 : {avg_s2:.1f} ms/sample  (only on {stage2_called} samples)")

    print(f"\n{'Function':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Total':>8}")
    print("-" * 70)

    per_function = {}
    for func_name in function_names + ["no_function"]:
        stats = results_by_func[func_name]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        total_func = stats["total"]
        prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{func_name:<25} {prec:>9.1f}% {rec:>9.1f}% {f1:>9.1f}% {total_func:>8}")
        per_function[func_name] = {"tp": tp, "fp": fp, "fn": fn, "total": total_func,
                                   "precision": prec, "recall": rec, "f1": f1}

    print("=" * 70)

    # Compare with Run 5 standalone
    run5_file = RESULTS_DIR / "peft_run5_ckpt800_results.json"
    if run5_file.exists():
        run5 = json.loads(run5_file.read_text())
        print(f"\nComparison vs Run 5 standalone ({run5['accuracy']:.1f}%):")
        delta = correct/total*100 - run5["accuracy"]
        sign = "+" if delta >= 0 else ""
        print(f"  Run 6 accuracy: {correct/total*100:.1f}%  ({sign}{delta:.1f}pp)")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "run_id": 6,
        "description": f"Two-stage: ML Stage1 + PEFT run{STAGE2_RUN_ID} ckpt{STAGE2_CHECKPOINT}",
        "stage1_backend": classifier.backend,
        "stage2_run_id": STAGE2_RUN_ID,
        "stage2_checkpoint": STAGE2_CHECKPOINT,
        "accuracy": correct / total * 100,
        "total_samples": total,
        "correct": correct,
        "stage1_blocked": stage1_blocked,
        "stage2_called": stage2_called,
        "stage1_fp": stage1_fp,
        "stage1_fn": stage1_fn,
        "avg_latency_s1_ms": avg_s1,
        "avg_latency_s2_ms": avg_s2,
        "per_function": per_function,
    }
    out_path = RESULTS_DIR / "peft_run6_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
