"""Weather information function."""

from src.functions.registry import FunctionDefinition, FunctionRegistry

SCHEMA = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "City or location name, e.g. Tokyo, New York",
        },
        "forecast_type": {
            "type": "string",
            "enum": ["current", "today", "week"],
            "description": "Type of weather forecast requested",
        },
    },
    "required": ["location"],
}


def weather_info_handler(
    location: str, forecast_type: str = "current"
) -> dict:
    """Mock handler for weather queries."""
    return {
        "function": "weather_info",
        "location": location,
        "forecast_type": forecast_type,
        "result": f"[Weather] {location} - {forecast_type}: "
        f"Weather information for the requested location.",
    }


def register(registry: FunctionRegistry):
    registry.register(
        FunctionDefinition(
            name="weather_info",
            description=(
                "Get current weather conditions or forecast for a specific location. "
                "Use when the user asks about weather, temperature, rain, snow, "
                "or climate conditions for a place."
            ),
            parameters=SCHEMA,
            handler=weather_info_handler,
        )
    )
