"""Central function registry and schema definitions."""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class FunctionDefinition:
    """A registered function with its schema and implementation."""

    name: str
    description: str
    parameters: dict  # JSON Schema for parameters
    handler: Callable  # The actual function to call

    def to_tool_schema(self) -> dict:
        """Convert to Ollama tool schema format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class FunctionRegistry:
    """Central registry for all callable functions."""

    def __init__(self):
        self._functions: dict[str, FunctionDefinition] = {}

    def register(self, func_def: FunctionDefinition):
        """Register a function definition."""
        self._functions[func_def.name] = func_def

    def get(self, name: str) -> Optional[FunctionDefinition]:
        """Get a function by name."""
        return self._functions.get(name)

    def get_all_tool_schemas(self) -> list[dict]:
        """Get all tool schemas for Ollama API."""
        return [f.to_tool_schema() for f in self._functions.values()]

    def get_all_names(self) -> list[str]:
        """Get all registered function names."""
        return list(self._functions.keys())

    def get_all_definitions(self) -> list[FunctionDefinition]:
        """Get all registered function definitions."""
        return list(self._functions.values())

    def execute(self, name: str, arguments: dict) -> dict:
        """Execute a registered function by name."""
        func_def = self._functions.get(name)
        if not func_def:
            return {"error": f"Unknown function: {name}"}
        try:
            return func_def.handler(**arguments)
        except TypeError as e:
            return {"error": f"Invalid arguments for {name}: {e}"}
