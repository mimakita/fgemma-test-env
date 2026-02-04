"""Shopping intent function - purchase intent, products, services, ads."""

from src.functions.registry import FunctionDefinition, FunctionRegistry

SCHEMA = {
    "type": "object",
    "properties": {
        "product_or_service": {
            "type": "string",
            # "description": "Product or service the user is interested in",
            "description": "ユーザーが興味のある商品やサービス",
        },
        "intent_type": {
            "type": "string",
            "enum": ["buy", "compare", "research", "deal"],
            # "description": "Type of shopping intent detected",
            "description": "検出された購買意図の種類",
        },
    },
    "required": ["product_or_service"],
}


def shopping_intent_handler(
    product_or_service: str, intent_type: str = "research"
) -> dict:
    """Mock handler for shopping intent queries."""
    return {
        "function": "shopping_intent",
        "product_or_service": product_or_service,
        "intent_type": intent_type,
        "result": f"[Shopping] {product_or_service} - {intent_type}: "
        f"Product/service information and recommendations available.",
    }


def register(registry: FunctionRegistry):
    registry.register(
        FunctionDefinition(
            name="shopping_intent",
            description=(
                # "Detect purchase intent and provide product or service information, "
                # "recommendations, or advertisements. Use when the user wants to buy, "
                # "compare, or research products or services, or shows commercial intent."
                "購買意図を検出し、商品やサービスの情報、おすすめ、広告を提供する。"
                "ユーザーが商品を買いたい、比較したい、調べたい時、または商業的な意図を示した時に使用する。"
            ),
            parameters=SCHEMA,
            handler=shopping_intent_handler,
        )
    )
