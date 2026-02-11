"""Run complete LoRA fine-tuning pipeline.

Orchestrates the full workflow:
1. Generate additional test data (if needed)
2. Split data (80:20)
3. Fine-tune with LoRA
4. Deploy to Ollama
5. Evaluate

Usage:
    python -m tools.run_pipeline --run-id 1 --seed 42
    python -m tools.run_pipeline --run-id 2 --seed 123
    python -m tools.run_pipeline --run-id 1 --seed 42 --skip-generate
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
FINETUNE_DIR = PROJECT_ROOT / "data" / "finetune"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def run_step(name: str, cmd: list[str], timeout: int = None) -> bool:
    """Run a pipeline step with logging."""
    print("\n" + "=" * 60)
    print(f"STEP: {name}")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}\n")

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            timeout=timeout,
        )
        elapsed = time.time() - start_time
        if result.returncode == 0:
            logger.info(f"Step '{name}' completed in {elapsed:.1f}s")
            return True
        else:
            logger.error(f"Step '{name}' failed with code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"Step '{name}' timed out after {timeout}s")
        return False
    except KeyboardInterrupt:
        logger.warning(f"Step '{name}' interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Step '{name}' error: {e}")
        return False


def check_data_ready() -> bool:
    """Check if test data is ready (doubled to 1900)."""
    combined_file = PROJECT_ROOT / "data" / "test" / "all_test_data.json"
    if not combined_file.exists():
        return False

    with open(combined_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = len(data)
    logger.info(f"Current test data count: {count}")

    # Target: 7 functions * 200 + 500 no_function = 1900
    if count >= 1900:
        return True

    logger.warning(f"Need at least 1900 cases, have {count}")
    return False


def get_venv_python(venv_name: str) -> str:
    """Get Python path for a virtual environment."""
    venv_python = PROJECT_ROOT / venv_name / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def main():
    parser = argparse.ArgumentParser(description="Run LoRA fine-tuning pipeline")
    parser.add_argument("--run-id", type=int, required=True, help="Run ID (1 or 2)")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--skip-generate", action="store_true", help="Skip data generation")
    parser.add_argument("--skip-finetune", action="store_true", help="Skip fine-tuning")
    parser.add_argument("--skip-deploy", action="store_true", help="Skip Ollama deployment")
    parser.add_argument("--iters", type=int, default=1000, help="Training iterations")
    args = parser.parse_args()

    run_dir = FINETUNE_DIR / f"run_{args.run_id}"
    results_run_dir = RESULTS_DIR / f"run_{args.run_id}"

    print("\n" + "=" * 60)
    print(f"FuncGemma LoRA Fine-tuning Pipeline")
    print(f"Run ID: {args.run_id}, Seed: {args.seed}")
    print("=" * 60)

    # Step 0: Check/Generate data
    if not args.skip_generate:
        if not check_data_ready():
            logger.info("Generating additional test data...")
            main_python = get_venv_python(".venv")
            success = run_step(
                "Generate Test Data",
                [main_python, "-m", "tools.generate_additional_data"],
                timeout=14400,  # 4 hours
            )
            if not success:
                logger.error("Data generation failed!")
                sys.exit(1)
        else:
            logger.info("Test data is ready (>=1900 cases)")

    # Step 1: Split data
    main_python = get_venv_python(".venv")
    success = run_step(
        "Split Data",
        [main_python, "-m", "tools.split_data",
         "--run-id", str(args.run_id),
         "--seed", str(args.seed)],
    )
    if not success:
        logger.error("Data split failed!")
        sys.exit(1)

    # Step 2: Fine-tune
    if not args.skip_finetune:
        ft_python = get_venv_python(".venv-ft")
        success = run_step(
            "LoRA Fine-tuning",
            [ft_python, "-m", "tools.finetune_lora",
             "--run-id", str(args.run_id),
             "--iters", str(args.iters)],
            timeout=7200,  # 2 hours
        )
        if not success:
            logger.error("Fine-tuning failed!")
            sys.exit(1)

    # Step 3: Deploy to Ollama
    if not args.skip_deploy:
        ft_python = get_venv_python(".venv-ft")
        success = run_step(
            "Deploy to Ollama",
            [ft_python, "-m", "tools.deploy_to_ollama",
             "--run-id", str(args.run_id)],
            timeout=600,  # 10 minutes
        )
        if not success:
            logger.error("Deployment failed!")
            sys.exit(1)

    # Step 4: Evaluate
    model_name = f"functiongemma-ft-run{args.run_id}"
    test_data_dir = str(run_dir)  # Use split test data

    main_python = get_venv_python(".venv")
    success = run_step(
        "Evaluate Fine-tuned Model",
        [main_python, "-m", "tools.evaluate",
         "--model", model_name,
         "--data-dir", test_data_dir,
         "--output-dir", str(results_run_dir)],
        timeout=3600,  # 1 hour
    )
    if not success:
        logger.error("Evaluation failed!")
        sys.exit(1)

    # Summary
    print("\n" + "=" * 60)
    print(f"Pipeline Complete (Run {args.run_id})")
    print("=" * 60)

    # Load and display results
    latest_metrics = results_run_dir / "latest_metrics.json"
    if latest_metrics.exists():
        with open(latest_metrics, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        overall = metrics.get("overall", {})
        print(f"\nResults:")
        print(f"  Model:             {model_name}")
        print(f"  Accuracy:          {overall.get('accuracy', 0):.1%}")
        print(f"  False Positive Rate: {overall.get('false_positive_rate', 0):.1%}")
        print(f"  Total Test Cases:  {overall.get('total_cases', 0)}")

    print(f"\nArtifacts:")
    print(f"  Train/Valid data:  {run_dir}")
    print(f"  Adapters:          {PROJECT_ROOT / 'data' / 'adapters' / f'run_{args.run_id}'}")
    print(f"  Fused model:       {PROJECT_ROOT / 'data' / 'fused' / f'run_{args.run_id}'}")
    print(f"  Results:           {results_run_dir}")

    # Save pipeline metadata
    pipeline_meta = {
        "run_id": args.run_id,
        "seed": args.seed,
        "iters": args.iters,
        "model_name": model_name,
        "metrics": metrics if latest_metrics.exists() else None,
    }
    meta_file = run_dir / "pipeline_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(pipeline_meta, f, ensure_ascii=False, indent=2)

    print(f"\nNext steps:")
    print(f"  - Compare with baseline: python -m tools.evaluate")
    print(f"  - Run second experiment: python -m tools.run_pipeline --run-id 2 --seed 123")


if __name__ == "__main__":
    main()
