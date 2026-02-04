"""Schedule and reminder function."""

from src.functions.registry import FunctionDefinition, FunctionRegistry

SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["create", "query", "cancel"],
            # "description": "Action to take: create, query, or cancel a schedule/reminder",
            "description": "実行するアクション: スケジュール/リマインダーの作成、確認、キャンセル",
        },
        "description": {
            "type": "string",
            # "description": "Description of the event or reminder",
            "description": "イベントやリマインダーの説明",
        },
        "datetime": {
            "type": "string",
            # "description": "Date and/or time for the event (ISO 8601 or natural language)",
            "description": "イベントの日時（ISO 8601形式または自然言語）",
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
                # "Create, manage, or query schedule events, reminders, and "
                # "time-related tasks. Use when the user wants to set a reminder, "
                # "schedule an event, check their calendar, or manage appointments."
                "スケジュールのイベント、リマインダー、時間関連のタスクを作成・管理・確認する。"
                "ユーザーがリマインダーを設定したい、予定を入れたい、カレンダーを確認したい、約束を管理したい時に使用する。"
            ),
            parameters=SCHEMA,
            handler=schedule_reminder_handler,
        )
    )
