"""Common utilities for FuncGemma tools.

Shared functions for data handling, file I/O, and logging configuration.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TEST_DATA_DIR = DATA_DIR / "test"
FINETUNE_DIR = DATA_DIR / "finetune"
PEFT_ADAPTERS_DIR = DATA_DIR / "peft_adapters"
RESULTS_DIR = DATA_DIR / "results"


def setup_logging(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Setup and return a configured logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger(name)


def extract_json_from_response(text: str) -> Optional[list]:
    """Extract JSON array from LLM response, handling various formats."""
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("items", "data", "test_data", "cases", "results", "conversations"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            if "conversation" in data:
                return [data]
        return None
    except json.JSONDecodeError:
        pass

    # Try finding JSON array in text
    bracket_start = text.find('[')
    bracket_end = text.rfind(']')
    if bracket_start >= 0 and bracket_end > bracket_start:
        try:
            data = json.loads(text[bracket_start:bracket_end + 1])
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    return None


def load_json(filepath: Path) -> list | dict:
    """Load JSON from file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list | dict, filepath: Path, ensure_dir: bool = True):
    """Save data to JSON file."""
    if ensure_dir:
        filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_jsonl(filepath: Path) -> list[dict]:
    """Load JSONL file."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], filepath: Path, ensure_dir: bool = True):
    """Save data to JSONL file."""
    if ensure_dir:
        filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def add_test_ids(cases: list[dict], prefix: str, start_id: int = 1) -> list[dict]:
    """Add unique test IDs to each case."""
    for i, case in enumerate(cases):
        case["test_id"] = f"{prefix}_{start_id + i:04d}"
        case["category"] = prefix
    return cases


def validate_test_case(case: dict, func_name: Optional[str], all_func_names: list[str]) -> bool:
    """Validate a single test case."""
    if not isinstance(case, dict):
        return False

    conv = case.get("conversation")
    if not isinstance(conv, list) or len(conv) < 1:
        return False

    for msg in conv:
        if not isinstance(msg, dict):
            return False
        if "role" not in msg or "content" not in msg:
            return False
        if msg["role"] not in ("user", "assistant"):
            return False
        if not msg["content"] or not msg["content"].strip():
            return False

    if not any(m["role"] == "user" for m in conv):
        return False

    expected = case.get("expected_function")
    if func_name is not None:
        if expected != func_name:
            return False
    else:
        if expected is not None and expected not in (None, "null", "none", ""):
            return False

    return True


def format_functiongemma_prompt(messages: list[dict], tools: list[dict]) -> str:
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
