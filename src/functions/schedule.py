"""Schedule and reminder function."""

from src.functions.registry import FunctionDefinition, FunctionRegistry

SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["create", "query", "cancel"],
            "description": "Action to take: create, query, or cancel a schedule/reminder",
        },
        "description": {
            "type": "string",
            "description": "Description of the event or reminder",
        },
        "datetime": {
            "type": "string",
            "description": "Date and/or time for the event (ISO 8601 or natural language)",
        },
    },
    "required": ["action", "description"],
}


def schedule_reminder_handler(
    action: str, description: str, datetime: str = ""
) -> dict:
    """Mock handler for schedule/reminder queries."""
    return {
        "function": "schedule_reminder",
        "action": action,
        "description": description,
        "datetime": datetime,
        "result": f"[Schedule] {action}: {description}"
        + (f" at {datetime}" if datetime else ""),
    }


def register(registry: FunctionRegistry):
    registry.register(
        FunctionDefinition(
            name="schedule_reminder",
            description=(
                "Create, manage, or query schedule events, reminders, and "
                "time-related tasks. Use when the user wants to set a reminder, "
                "schedule an event, check their calendar, or manage appointments."
            ),
            parameters=SCHEMA,
            handler=schedule_reminder_handler,
        )
    )
