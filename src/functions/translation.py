"""Translation assistance function."""

from src.functions.registry import FunctionDefinition, FunctionRegistry

SCHEMA = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
            # "description": "Text to translate",
            "description": "翻訳するテキスト",
        },
        "source_language": {
            "type": "string",
            # "description": "Source language (e.g. Japanese, English). Auto-detect if not specified.",
            "description": "翻訳元の言語（例: 日本語、英語）。未指定の場合は自動検出",
        },
        "target_language": {
            "type": "string",
            # "description": "Target language for translation (e.g. English, Japanese)",
            "description": "翻訳先の言語（例: 英語、日本語）",
        },
    },
    "required": ["text", "target_language"],
}


def translation_assist_handler(
    text: str, target_language: str, source_language: str = "auto"
) -> dict:
    """Mock handler for translation queries."""
    return {
        "function": "translation_assist",
        "text": text[:100],
        "source_language": source_language,
        "target_language": target_language,
        "result": f"[Translation] {source_language} -> {target_language}: "
        f"Translation result for the given text.",
    }


def register(registry: FunctionRegistry):
    registry.register(
        FunctionDefinition(
            name="translation_assist",
            description=(
                # "Translate text between languages or provide language-related "
                # "assistance. Use when the user asks to translate something, "
                # "wants to know how to say something in another language, "
                # "or needs language help."
                "言語間のテキスト翻訳や言語関連の支援を行う。"
                "ユーザーが何かを翻訳したい、別の言語での言い方を知りたい、言語の助けが必要な時に使用する。"
            ),
            parameters=SCHEMA,
            handler=translation_assist_handler,
        )
    )
