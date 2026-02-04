"""Celebrity information function - famous and notable people."""

from src.functions.registry import FunctionDefinition, FunctionRegistry

SCHEMA = {
    "type": "object",
    "properties": {
        "person_name": {
            "type": "string",
            # "description": "Name of the famous or notable person",
            "description": "有名人や著名人の名前",
        },
        "info_type": {
            "type": "string",
            "enum": ["biography", "career", "achievements", "general"],
            # "description": "Type of information requested about the person",
            "description": "その人物についてリクエストされた情報の種類",
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
                # "Get information about a famous or notable person, including their "
                # "biography, career, and achievements. Use when the user asks about "
                # "a celebrity, public figure, historical person, or notable individual."
                "有名人や著名人の経歴、キャリア、功績などの情報を取得する。"
                "ユーザーが芸能人、公人、歴史上の人物、著名人について尋ねた時に使用する。"
            ),
            parameters=SCHEMA,
            handler=celebrity_info_handler,
        )
    )
