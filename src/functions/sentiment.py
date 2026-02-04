"""Sentiment label function - emotion and sentiment analysis."""

from src.functions.registry import FunctionDefinition, FunctionRegistry

SCHEMA = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
            "description": "Text to analyze for sentiment or emotion",
        },
        "granularity": {
            "type": "string",
            "enum": ["basic", "detailed"],
            "description": (
                "Level of sentiment detail: basic (positive/negative/neutral) "
                "or detailed (joy, anger, sadness, fear, surprise, disgust)"
            ),
        },
    },
    "required": ["text"],
}


def sentiment_label_handler(
    text: str, granularity: str = "basic"
) -> dict:
    """Mock handler for sentiment analysis queries."""
    return {
        "function": "sentiment_label",
        "text": text[:100],  # Truncate for display
        "granularity": granularity,
        "result": f"[Sentiment] granularity={granularity}: "
        f"Analysis result for the given text.",
    }


def register(registry: FunctionRegistry):
    registry.register(
        FunctionDefinition(
            name="sentiment_label",
            description=(
                "Analyze and label the sentiment or emotion expressed in text or "
                "conversation. Use when the user asks about feelings, emotions, "
                "mood, or sentiment of a text, review, or conversation."
            ),
            parameters=SCHEMA,
            handler=sentiment_label_handler,
        )
    )
