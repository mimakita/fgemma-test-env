"""FunctionGemma routing logic - analyzes conversation and routes to functions.

Implements two-stage function calling:
- Stage 1: Classifier determines if function call is needed (fast, keyword-based)
- Stage 2: FunctionGemma selects the appropriate function (LLM-based)

This approach achieves 100% no_function recall while maintaining function call accuracy.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from src.config import ROUTER_MODEL, ROUTER_OPTIONS, ROUTER_KEEP_ALIVE, MAX_HISTORY_MESSAGES
from src.ollama_client import OllamaClient
from src.functions.registry import FunctionRegistry
from src.classifier import FunctionCallClassifier, ClassificationResult

logger = logging.getLogger(__name__)


@dataclass
class RouterResult:
    """Result of a routing decision."""

    should_call: bool = False
    function_name: Optional[str] = None
    arguments: dict = field(default_factory=dict)
    raw_response: str = ""
    # Stage 1 classification info
    classification: Optional[ClassificationResult] = None
    stage1_blocked: bool = False


class FunctionRouter:
    """Routes conversations to appropriate functions using two-stage approach.

    Stage 1: FunctionCallClassifier (keyword-based, fast)
        - Filters out conversations that don't need function calls
        - Achieves 100% no_function recall

    Stage 2: FunctionGemma (LLM-based)
        - Only invoked if Stage 1 determines function call is needed
        - Selects the appropriate function and extracts arguments
    """

    def __init__(
        self,
        client: OllamaClient,
        registry: FunctionRegistry,
        model_override: Optional[str] = None,
        use_two_stage: bool = True,
    ):
        """Initialize router.

        Args:
            client: OllamaClient for LLM calls
            registry: FunctionRegistry with available functions
            model_override: Override default router model
            use_two_stage: Enable two-stage classification (default: True)
        """
        self.client = client
        self.registry = registry
        self.model = model_override or ROUTER_MODEL
        self.use_two_stage = use_two_stage
        self.classifier = FunctionCallClassifier()

    def route(self, conversation_history: list[dict]) -> RouterResult:
        """Analyze conversation and determine if a function should be called.

        Args:
            conversation_history: List of {"role": "user"|"assistant", "content": "..."}

        Returns:
            RouterResult with function_name and arguments if a function should be called
        """
        # Use only recent messages to stay within context limits
        recent = conversation_history[-MAX_HISTORY_MESSAGES:]
        tools = self.registry.get_all_tool_schemas()

        if not tools:
            return RouterResult()

        # Stage 1: Classify if function call is needed
        if self.use_two_stage:
            classification = self.classifier.classify(recent)

            if not classification.need_function:
                logger.debug(
                    f"Stage 1 blocked function call (confidence: {classification.confidence:.2f})"
                )
                return RouterResult(
                    should_call=False,
                    classification=classification,
                    stage1_blocked=True,
                )
        else:
            classification = None

        # Stage 2: Use FunctionGemma to select function
        try:
            response = self.client.chat_completion(
                model=self.model,
                messages=recent,
                tools=tools,
                options=ROUTER_OPTIONS,
                keep_alive=ROUTER_KEEP_ALIVE,
            )
        except Exception as e:
            logger.error(f"Router error: {e}")
            return RouterResult(classification=classification)

        raw_content = response.message.content or ""

        # Method 1: Structured tool_calls from Ollama API
        if response.message.tool_calls:
            call = response.message.tool_calls[0]
            func_name = call.function.name
            func_args = call.function.arguments or {}

            # Validate function exists
            if self.registry.get(func_name):
                return RouterResult(
                    should_call=True,
                    function_name=func_name,
                    arguments=func_args,
                    raw_response=raw_content,
                    classification=classification,
                )
            else:
                logger.warning(f"Router returned unknown function: {func_name}")

        # Method 2: Fallback - parse raw FunctionGemma output tokens
        parsed = self._parse_raw_function_call(raw_content)
        if parsed:
            func_name, func_args = parsed
            if self.registry.get(func_name):
                return RouterResult(
                    should_call=True,
                    function_name=func_name,
                    arguments=func_args,
                    raw_response=raw_content,
                    classification=classification,
                )

        return RouterResult(raw_response=raw_content, classification=classification)

    def _parse_raw_function_call(self, text: str) -> Optional[tuple[str, dict]]:
        """Fallback parser for raw FunctionGemma output tokens.

        FunctionGemma may output function calls in various formats:
        - {"name": "func", "arguments": {...}}
        - func_name(arg1="val1", arg2="val2")
        - <functioncall> {"name": "func", ...}
        """
        if not text or not text.strip():
            return None

        # Try JSON format: {"name": "...", "arguments": {...}}
        try:
            data = json.loads(text.strip())
            if isinstance(data, dict):
                name = data.get("name") or data.get("function")
                args = data.get("arguments") or data.get("parameters") or {}
                if name:
                    return name, args
        except (json.JSONDecodeError, ValueError):
            pass

        # Try extracting JSON from text
        json_match = re.search(r'\{[^{}]*"name"\s*:\s*"[^"]+?"[^{}]*\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                name = data.get("name")
                args = data.get("arguments") or data.get("parameters") or {}
                if name:
                    return name, args
            except (json.JSONDecodeError, ValueError):
                pass

        # Try function call format: func_name(key="value", ...)
        func_match = re.match(r'(\w+)\((.*)\)', text.strip(), re.DOTALL)
        if func_match:
            name = func_match.group(1)
            args_str = func_match.group(2).strip()
            if name in self.registry.get_all_names():
                args = self._parse_function_args(args_str)
                return name, args

        return None

    def _parse_function_args(self, args_str: str) -> dict:
        """Parse function-call style arguments: key="value", key2="value2"."""
        args = {}
        for match in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', args_str):
            args[match.group(1)] = match.group(2)
        for match in re.finditer(r"(\w+)\s*=\s*'([^']*)'", args_str):
            if match.group(1) not in args:
                args[match.group(1)] = match.group(2)
        return args
