"""FunctionGemma routing logic - analyzes conversation and routes to functions."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from src.config import ROUTER_MODEL, ROUTER_OPTIONS, ROUTER_KEEP_ALIVE, MAX_HISTORY_MESSAGES
from src.ollama_client import OllamaClient
from src.functions.registry import FunctionRegistry

logger = logging.getLogger(__name__)


@dataclass
class RouterResult:
    """Result of a routing decision."""

    should_call: bool = False
    function_name: Optional[str] = None
    arguments: dict = field(default_factory=dict)
    raw_response: str = ""


class FunctionRouter:
    """Routes conversations to appropriate functions using FunctionGemma."""

    def __init__(
        self,
        client: OllamaClient,
        registry: FunctionRegistry,
        model_override: Optional[str] = None,
    ):
        self.client = client
        self.registry = registry
        self.model = model_override or ROUTER_MODEL

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
            return RouterResult()

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
                )

        return RouterResult(raw_response=raw_content)

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
