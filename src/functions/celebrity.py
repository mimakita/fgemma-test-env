"""Celebrity information function - famous and notable people."""

from src.functions.registry import FunctionDefinition, FunctionRegistry

SCHEMA = {
    "type": "object",
    "properties": {
        "person_name": {
            "type": "string",
            "description": "Name of the famous or notable person",
        },
        "info_type": {
            "type": "string",
            "enum": ["biography", "career", "achievements", "general"],
            "description": "Type of information requested about the person",
        },
    },
    "required": ["person_name"],
}


def celebrity_info_handler(
    person_name: str, info_type: str = "general"
) -> dict:
    """Mock handler for celebrity information queries."""
    return {
        "function": "celebrity_info",
        "person_name": person_name,
        "info_type": info_type,
        "result": f"[Celebrity Info] {person_name} - {info_type}: "
        f"Notable figure with significant contributions.",
    }


def register(registry: FunctionRegistry):
    registry.register(
        FunctionDefinition(
            name="celebrity_info",
            description=(
                "Get information about a famous or notable person, including their "
                "biography, career, and achievements. Use when the user asks about "
                "a celebrity, public figure, historical person, or notable individual."
            ),
            parameters=SCHEMA,
            handler=celebrity_info_handler,
        )
    )
