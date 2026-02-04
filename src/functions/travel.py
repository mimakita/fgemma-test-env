"""Travel guide function - place names and travel information."""

from src.functions.registry import FunctionDefinition, FunctionRegistry

SCHEMA = {
    "type": "object",
    "properties": {
        "destination": {
            "type": "string",
            # "description": "The place or destination name, e.g. Tokyo, Paris, Kyoto",
            "description": "場所や目的地の名前。例: 東京、パリ、京都",
        },
        "info_type": {
            "type": "string",
            "enum": ["attractions", "tips", "transportation", "food", "general"],
            # "description": "Type of travel information requested",
            "description": "リクエストされた旅行情報の種類",
        },
    },
    "required": ["destination"],
}


def travel_guide_handler(
    destination: str, info_type: str = "general"
) -> dict:
    """Mock handler for travel guide queries."""
    return {
        "function": "travel_guide",
        "destination": destination,
        "info_type": info_type,
        "result": f"[Travel Info] {destination} - {info_type}: "
        f"Popular destination with many things to explore.",
    }


def register(registry: FunctionRegistry):
    registry.register(
        FunctionDefinition(
            name="travel_guide",
            description=(
                # "Get travel information, tourist attractions, tips, and guides "
                # "for a specific place or destination. Use when the user asks about "
                # "a place to visit, sightseeing, travel plans, or tourism."
                "特定の場所や目的地の旅行情報、観光名所、ヒント、ガイドを取得する。"
                "ユーザーが訪問先、観光、旅行計画、ツーリズムについて尋ねた時に使用する。"
            ),
            parameters=SCHEMA,
            handler=travel_guide_handler,
        )
    )
