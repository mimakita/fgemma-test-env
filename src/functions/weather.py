"""Weather information function."""

from src.functions.registry import FunctionDefinition, FunctionRegistry

SCHEMA = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            # "description": "City or location name, e.g. Tokyo, New York",
            "description": "都市名や場所の名前。例: 東京、ニューヨーク",
        },
        "forecast_type": {
            "type": "string",
            "enum": ["current", "today", "week"],
            # "description": "Type of weather forecast requested",
            "description": "リクエストされた天気予報の種類",
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
                # "Get current weather conditions or forecast for a specific location. "
                # "Use when the user asks about weather, temperature, rain, snow, "
                # "or climate conditions for a place."
                "特定の場所の現在の天気状況や天気予報を取得する。"
                "ユーザーが天気、気温、雨、雪、気候条件について尋ねた時に使用する。"
            ),
            parameters=SCHEMA,
            handler=weather_info_handler,
        )
    )
