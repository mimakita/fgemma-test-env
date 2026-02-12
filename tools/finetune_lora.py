"""LoRA Fine-tuning script using mlx-lm.

Fine-tunes FunctionGemma (270M) with LoRA adapters on Apple Silicon.

Usage:
    python -m tools.finetune_lora --run-id 1
    python -m tools.finetune_lora --run-id 1 --iters 500 --lr 5e-5

Requires: .venv-ft (Python 3.12 with mlx-lm installed)
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
FINETUNE_DIR = PROJECT_ROOT / "data" / "finetune"
ADAPTERS_DIR = PROJECT_ROOT / "data" / "adapters"

# Default hyperparameters
DEFAULT_MODEL = "google/functiongemma-270m-it"
DEFAULT_ITERS = 200  # Reduced from 1000 to prevent overfitting
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 5e-5  # Reduced from 1e-4 for more stable learning
DEFAULT_LORA_LAYERS = 4  # Reduced from 8 to limit capacity
DEFAULT_LORA_RANK = 4  # Reduced from 8 to limit capacity
DEFAULT_MAX_SEQ_LENGTH = 512


def find_venv_python() -> str:
    """Find the Python 3.12 venv with mlx-lm installed."""
    venv_ft = PROJECT_ROOT / ".venv-ft" / "bin" / "python"
    if venv_ft.exists():
        # Verify mlx-lm is installed
        result = subprocess.run(
            [str(venv_ft), "-c", "import mlx_lm; print(mlx_lm.__version__)"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info(f"Using .venv-ft Python with mlx-lm {result.stdout.strip()}")
            return str(venv_ft)

    # Fallback: current Python
    try:
        import mlx_lm
        logger.info(f"Using current Python with mlx-lm {mlx_lm.__version__}")
        return sys.executable
    except ImportError:
        pass

    logger.error(
        "mlx-lm not found! Please install:\n"
        "  python3.12 -m venv .venv-ft\n"
        "  .venv-ft/bin/pip install mlx-lm transformers"
    )
    sys.exit(1)


def create_lora_config(args, run_dir: Path) -> Path:
    """Create LoRA training configuration file."""
    config = {
        "model": args.model,
        "train": True,
        "data": str(run_dir),
        "adapter_path": str(ADAPTERS_DIR / f"run_{args.run_id}"),
        "iters": args.iters,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "lora_layers": args.lora_layers,
        "lora_parameters": {
            "rank": args.lora_rank,
            "alpha": args.lora_rank * 2,
            "dropout": 0.05,
            "scale": 1.0,
        },
        "max_seq_length": args.max_seq_length,
        "grad_checkpoint": True,
        "steps_per_report": 50,
        "steps_per_eval": 100,
        "save_every": 200,
    }

    config_path = run_dir / "lora_config.yaml"
    # Write as YAML-like JSON (mlx-lm accepts both)
    config_path_json = run_dir / "lora_config.json"
    with open(config_path_json, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Config saved to {config_path_json}")
    return config_path_json


def run_finetune(python_path: str, args) -> bool:
    """Run mlx-lm LoRA fine-tuning."""
    run_dir = FINETUNE_DIR / f"run_{args.run_id}"
    adapter_dir = ADAPTERS_DIR / f"run_{args.run_id}"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Check training data exists
    train_file = run_dir / "train.jsonl"
    valid_file = run_dir / "valid.jsonl"
    if not train_file.exists():
        logger.error(f"Training data not found: {train_file}")
        logger.error("Run split_data.py first!")
        return False

    # Count training samples
    with open(train_file, "r", encoding="utf-8") as f:
        train_count = sum(1 for _ in f)
    logger.info(f"Training samples: {train_count}")

    if valid_file.exists():
        with open(valid_file, "r", encoding="utf-8") as f:
            valid_count = sum(1 for _ in f)
        logger.info(f"Validation samples: {valid_count}")

    # Build mlx_lm lora command (new format: python -m mlx_lm lora ...)
    cmd = [
        python_path, "-m", "mlx_lm", "lora",
        "--model", args.model,
        "--data", str(run_dir),
        "--adapter-path", str(adapter_dir),
        "--train",
        "--iters", str(args.iters),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.lr),
        "--num-layers", str(args.lora_layers),
        "--max-seq-length", str(args.max_seq_length),
        "--grad-checkpoint",
        "--steps-per-report", "50",
        "--steps-per-eval", "100",
        "--save-every", "200",
    ]

    # Add validation file if exists
    if valid_file.exists():
        cmd.extend(["--val-batches", "25"])

    logger.info(f"Running: {' '.join(cmd)}")
    print("\n" + "=" * 60)
    print(f"Starting LoRA Fine-tuning (Run {args.run_id})")
    print("=" * 60)
    print(f"  Model:          {args.model}")
    print(f"  Training data:  {train_file}")
    print(f"  Adapter output: {adapter_dir}")
    print(f"  Iterations:     {args.iters}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  LoRA layers:    {args.lora_layers}")
    print(f"  LoRA rank:      {args.lora_rank}")
    print(f"  Max seq length: {args.max_seq_length}")
    print("=" * 60 + "\n")

    # Run training
    try:
        process = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            check=True,
        )
        logger.info("Fine-tuning completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Fine-tuning failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        logger.warning("Fine-tuning interrupted by user")
        return False


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tune FunctionGemma")
    parser.add_argument("--run-id", type=int, required=True, help="Run ID (1 or 2)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HuggingFace model ID")
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS, help="Training iterations")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--lora-layers", type=int, default=DEFAULT_LORA_LAYERS, help="Number of LoRA layers")
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK, help="LoRA rank")
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH, help="Max sequence length")
    args = parser.parse_args()

    # Find Python with mlx-lm
    python_path = find_venv_python()

    # Save config
    run_dir = FINETUNE_DIR / f"run_{args.run_id}"
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        logger.error("Run split_data.py first!")
        sys.exit(1)

    create_lora_config(args, run_dir)

    # Run fine-tuning
    success = run_finetune(python_path, args)

    if success:
        adapter_dir = ADAPTERS_DIR / f"run_{args.run_id}"
        print("\n" + "=" * 60)
        print(f"Fine-tuning Complete (Run {args.run_id})")
        print("=" * 60)
        print(f"  Adapters saved to: {adapter_dir}")
        print(f"  Next: python -m tools.deploy_to_ollama --run-id {args.run_id}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
