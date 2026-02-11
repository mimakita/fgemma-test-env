"""Deploy fine-tuned model to Ollama.

Fuses LoRA adapters with base model and creates an Ollama model.

Usage:
    python -m tools.deploy_to_ollama --run-id 1

Requires: .venv-ft (Python 3.12 with mlx-lm installed)
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
ADAPTERS_DIR = PROJECT_ROOT / "data" / "adapters"
FUSED_DIR = PROJECT_ROOT / "data" / "fused"

DEFAULT_MODEL = "google/functiongemma-270m-it"


def find_venv_python() -> str:
    """Find the Python 3.12 venv with mlx-lm installed."""
    venv_ft = PROJECT_ROOT / ".venv-ft" / "bin" / "python"
    if venv_ft.exists():
        result = subprocess.run(
            [str(venv_ft), "-c", "import mlx_lm; print(mlx_lm.__version__)"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return str(venv_ft)

    try:
        import mlx_lm
        return sys.executable
    except ImportError:
        pass

    logger.error("mlx-lm not found!")
    sys.exit(1)


def fuse_adapters(python_path: str, model: str, run_id: int) -> Path:
    """Fuse LoRA adapters with base model."""
    adapter_dir = ADAPTERS_DIR / f"run_{run_id}"
    fused_dir = FUSED_DIR / f"run_{run_id}"

    if not adapter_dir.exists():
        logger.error(f"Adapter directory not found: {adapter_dir}")
        logger.error("Run finetune_lora.py first!")
        sys.exit(1)

    # Check adapter files exist
    adapter_files = list(adapter_dir.glob("adapters*.safetensors"))
    if not adapter_files:
        # Also check for npz format
        adapter_files = list(adapter_dir.glob("adapters*.npz"))
    if not adapter_files:
        logger.error(f"No adapter files found in {adapter_dir}")
        sys.exit(1)

    logger.info(f"Found adapter files: {[f.name for f in adapter_files]}")

    # Fuse using mlx_lm.fuse
    cmd = [
        python_path, "-m", "mlx_lm.fuse",
        "--model", model,
        "--adapter-path", str(adapter_dir),
        "--save-path", str(fused_dir),
    ]

    logger.info(f"Fusing: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
        logger.info(f"Fused model saved to {fused_dir}")
        return fused_dir
    except subprocess.CalledProcessError as e:
        logger.error(f"Fuse failed with exit code {e.returncode}")
        sys.exit(1)


def convert_to_gguf(python_path: str, fused_dir: Path, run_id: int) -> Path:
    """Convert fused model to GGUF format for Ollama."""
    gguf_path = fused_dir / f"functiongemma-ft-run{run_id}.gguf"

    if gguf_path.exists():
        logger.info(f"GGUF already exists: {gguf_path}")
        return gguf_path

    # Try mlx_lm.convert with --gguf flag
    cmd = [
        python_path, "-m", "mlx_lm.convert",
        "--hf-path", str(fused_dir),
        "--mlx-path", str(fused_dir / "mlx_gguf"),
        "--quantize",
    ]

    logger.info("Attempting GGUF conversion via mlx_lm.convert...")

    try:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True, capture_output=True, text=True)
        # Check if GGUF was created
        gguf_files = list(fused_dir.glob("**/*.gguf"))
        if gguf_files:
            return gguf_files[0]
    except subprocess.CalledProcessError:
        logger.info("mlx_lm.convert GGUF not available, trying alternative...")

    # Alternative: Use llama.cpp convert if available
    llama_convert = Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"
    if llama_convert.exists():
        cmd = [
            sys.executable, str(llama_convert),
            str(fused_dir),
            "--outfile", str(gguf_path),
            "--outtype", "f16",
        ]
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"GGUF created: {gguf_path}")
            return gguf_path
        except subprocess.CalledProcessError:
            pass

    # If no GGUF, use safetensors directly with Modelfile FROM path
    logger.warning("GGUF conversion not available - will use safetensors with Ollama")
    return fused_dir


def create_ollama_model(fused_dir: Path, run_id: int) -> str:
    """Create Ollama model from fused weights."""
    model_name = f"functiongemma-ft-run{run_id}"

    # Check for GGUF file
    gguf_files = list(fused_dir.glob("**/*.gguf"))

    if gguf_files:
        model_source = str(gguf_files[0])
        logger.info(f"Using GGUF: {model_source}")
    else:
        # Use safetensors directory directly
        model_source = str(fused_dir)
        logger.info(f"Using safetensors directory: {model_source}")

    # Create Modelfile
    modelfile_content = f"""FROM {model_source}

PARAMETER temperature 0.0
PARAMETER num_ctx 4096
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<eos>"

TEMPLATE \"\"\"{{{{- if .Tools }}}}
Available tools:
{{{{- range .Tools }}}}
{{{{- . }}}}
{{{{- end }}}}
{{{{- end }}}}

{{{{- range .Messages }}}}
<start_of_turn>{{{{ .Role }}}}
{{{{ .Content }}}}
<end_of_turn>
{{{{- end }}}}
<start_of_turn>model
\"\"\"
"""

    # Write Modelfile
    modelfile_path = fused_dir / "Modelfile"
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    logger.info(f"Modelfile written to {modelfile_path}")

    # Create Ollama model
    cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
    logger.info(f"Creating Ollama model: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True, timeout=300)
        logger.info(f"Ollama model created: {model_name}")

        # Verify
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
        )
        if model_name in result.stdout:
            logger.info(f"Verified: {model_name} is available in Ollama")
        else:
            logger.warning(f"Model {model_name} not found in ollama list output")

        return model_name

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create Ollama model: {e}")
        # Try alternative: import from safetensors
        logger.info("Trying alternative import method...")
        return try_alternative_import(fused_dir, model_name, run_id)
    except subprocess.TimeoutExpired:
        logger.error("Ollama create timed out (5 min)")
        sys.exit(1)


def try_alternative_import(fused_dir: Path, model_name: str, run_id: int) -> str:
    """Try alternative methods to import model to Ollama."""
    # Method: Use ollama create with local path
    # Some Ollama versions support --from for local paths
    logger.info("Attempting direct safetensors import...")

    # Create a simpler Modelfile pointing to the directory
    modelfile_content = f"""FROM {fused_dir}

PARAMETER temperature 0.0
PARAMETER num_ctx 4096
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".Modelfile", delete=False
    ) as f:
        f.write(modelfile_content)
        tmp_modelfile = f.name

    try:
        subprocess.run(
            ["ollama", "create", model_name, "-f", tmp_modelfile],
            check=True,
            timeout=300,
        )
        logger.info(f"Alternative import succeeded: {model_name}")
        return model_name
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error(f"Alternative import also failed: {e}")
        logger.error(
            f"\nManual steps to deploy:\n"
            f"1. Convert to GGUF: install llama.cpp and run:\n"
            f"   python llama.cpp/convert_hf_to_gguf.py {fused_dir} --outfile {fused_dir}/model.gguf\n"
            f"2. Create Ollama model:\n"
            f"   ollama create {model_name} -f {fused_dir}/Modelfile\n"
        )
        sys.exit(1)
    finally:
        Path(tmp_modelfile).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Deploy fine-tuned model to Ollama")
    parser.add_argument("--run-id", type=int, required=True, help="Run ID (1 or 2)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base HuggingFace model ID")
    parser.add_argument("--skip-fuse", action="store_true", help="Skip fusing (use existing fused model)")
    args = parser.parse_args()

    fused_dir = FUSED_DIR / f"run_{args.run_id}"

    print("\n" + "=" * 60)
    print(f"Deploying Fine-tuned Model (Run {args.run_id})")
    print("=" * 60)

    if not args.skip_fuse:
        # Step 1: Fuse adapters
        python_path = find_venv_python()
        fused_dir = fuse_adapters(python_path, args.model, args.run_id)
    else:
        if not fused_dir.exists():
            logger.error(f"Fused directory not found: {fused_dir}")
            sys.exit(1)
        logger.info(f"Using existing fused model: {fused_dir}")

    # Step 2: Convert to GGUF (if possible)
    python_path = find_venv_python()
    convert_to_gguf(python_path, fused_dir, args.run_id)

    # Step 3: Create Ollama model
    model_name = create_ollama_model(fused_dir, args.run_id)

    print("\n" + "=" * 60)
    print(f"Deployment Complete (Run {args.run_id})")
    print("=" * 60)
    print(f"  Model name: {model_name}")
    print(f"  Test: ollama run {model_name} 'test'")
    print(f"  Evaluate: python -m tools.evaluate --model {model_name}")


if __name__ == "__main__":
    main()
