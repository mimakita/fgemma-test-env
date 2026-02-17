"""Generate diverse test data for all function domains.

This script generates additional test cases for each function to balance the dataset.
Each function has specific patterns to ensure diversity and clear intent signals.

Usage:
    python -m tools.generate_diverse_functions --function celebrity_info --count 150
    python -m tools.generate_diverse_functions --all --target 370
"""

import argparse
import json
import logging
import random
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "test"


# ============================================================
# CELEBRITY_INFO
# ============================================================

def generate_celebrity_info() -> dict:
    """Generate celebrity info lookup requests."""
    celebrities = [
        ("大谷翔平", "野球選手"),
        ("宮崎駿", "映画監督"),
        ("村上春樹", "作家"),
        ("イチロー", "元野球選手"),
        ("羽生結弦", "フィギュアスケーター"),
        ("米津玄師", "ミュージシャン"),
        ("北野武", "映画監督"),
        ("坂本龍一", "音楽家"),
        ("安藤忠雄", "建築家"),
        ("草間彌生", "アーティスト"),
        ("Taylor Swift", "singer"),
        ("Elon Musk", "entrepreneur"),
        ("BTS", "K-POPグループ"),
        ("渡辺直美", "タレント"),
        ("松本人志", "お笑い芸人"),
    ]

    name, role = random.choice(celebrities)

    patterns = [
        {
            "conversation": [
                {"role": "user", "content": f"{name}について教えてください。"},
            ],
            "info_type": "general",
        },
        {
            "conversation": [
                {"role": "user", "content": f"{name}のプロフィールを知りたいです。"},
            ],
            "info_type": "profile",
        },
        {
            "conversation": [
                {"role": "user", "content": f"{name}って誰ですか？"},
            ],
            "info_type": "general",
        },
        {
            "conversation": [
                {"role": "user", "content": f"{role}の{name}について詳しく教えて。"},
            ],
            "info_type": "general",
        },
        {
            "conversation": [
                {"role": "user", "content": f"{name}の経歴を調べてもらえますか？"},
            ],
            "info_type": "career",
        },
        {
            "conversation": [
                {"role": "user", "content": "有名人について質問があります。"},
                {"role": "assistant", "content": "どなたについて知りたいですか？"},
                {"role": "user", "content": f"{name}です。"},
            ],
            "info_type": "general",
        },
        {
            "conversation": [
                {"role": "user", "content": f"{name}の最近の活動は何ですか？"},
            ],
            "info_type": "recent_activity",
        },
        {
            "conversation": [
                {"role": "user", "content": f"{name}は何歳ですか？"},
            ],
            "info_type": "profile",
        },
    ]

    pattern = random.choice(patterns)
    return {
        "conversation": pattern["conversation"],
        "expected_function": "celebrity_info",
        "expected_arguments": {
            "name": name,
            "info_type": pattern["info_type"],
        },
    }


# ============================================================
# TRAVEL_GUIDE
# ============================================================

def generate_travel_guide() -> dict:
    """Generate travel guide requests."""
    destinations = [
        "京都", "東京", "大阪", "北海道", "沖縄", "奈良", "広島", "金沢",
        "箱根", "鎌倉", "日光", "富士山", "高山", "長崎", "福岡",
        "Paris", "New York", "London", "Seoul", "Bangkok", "Hawaii",
    ]

    info_types = [
        ("観光スポット", "attractions"),
        ("おすすめの食べ物", "food"),
        ("ホテル", "accommodation"),
        ("アクセス方法", "transportation"),
        ("おすすめの季節", "best_season"),
    ]

    dest = random.choice(destinations)
    info_ja, info_type = random.choice(info_types)

    patterns = [
        {
            "conversation": [
                {"role": "user", "content": f"{dest}の{info_ja}を教えてください。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": f"{dest}に旅行に行きたいのですが、おすすめはありますか？"},
            ],
            "info_type": "attractions",
        },
        {
            "conversation": [
                {"role": "user", "content": f"{dest}への行き方を教えてください。"},
            ],
            "info_type": "transportation",
        },
        {
            "conversation": [
                {"role": "user", "content": "旅行の計画を立てています。"},
                {"role": "assistant", "content": "どちらへ行かれる予定ですか？"},
                {"role": "user", "content": f"{dest}です。観光情報を教えてください。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": f"{dest}で有名な場所はどこですか？"},
            ],
            "info_type": "attractions",
        },
        {
            "conversation": [
                {"role": "user", "content": f"{dest}旅行のベストシーズンはいつですか？"},
            ],
            "info_type": "best_season",
        },
        {
            "conversation": [
                {"role": "user", "content": f"{dest}でおいしいものが食べたいです。"},
            ],
            "info_type": "food",
        },
    ]

    pattern = random.choice(patterns)
    return {
        "conversation": pattern["conversation"],
        "expected_function": "travel_guide",
        "expected_arguments": {
            "destination": dest,
            "info_type": pattern.get("info_type", info_type),
        },
    }


# ============================================================
# WEATHER_INFO
# ============================================================

def generate_weather_info() -> dict:
    """Generate weather info requests."""
    locations = [
        "東京", "大阪", "名古屋", "福岡", "札幌", "仙台", "横浜", "神戸",
        "京都", "広島", "New York", "London", "Paris", "Singapore",
    ]

    loc = random.choice(locations)

    patterns = [
        {
            "conversation": [
                {"role": "user", "content": f"{loc}の天気を教えてください。"},
            ],
            "forecast_type": "current",
        },
        {
            "conversation": [
                {"role": "user", "content": f"今日の{loc}の天気は？"},
            ],
            "forecast_type": "today",
        },
        {
            "conversation": [
                {"role": "user", "content": f"明日の{loc}の天気予報を教えて。"},
            ],
            "forecast_type": "tomorrow",
        },
        {
            "conversation": [
                {"role": "user", "content": f"{loc}は今雨が降っていますか？"},
            ],
            "forecast_type": "current",
        },
        {
            "conversation": [
                {"role": "user", "content": f"週末の{loc}の天気はどうですか？"},
            ],
            "forecast_type": "weekend",
        },
        {
            "conversation": [
                {"role": "user", "content": "天気を確認したいのですが。"},
                {"role": "assistant", "content": "どちらの地域ですか？"},
                {"role": "user", "content": f"{loc}です。"},
            ],
            "forecast_type": "current",
        },
        {
            "conversation": [
                {"role": "user", "content": f"{loc}の気温は何度ですか？"},
            ],
            "forecast_type": "current",
        },
        {
            "conversation": [
                {"role": "user", "content": f"{loc}は傘が必要ですか？"},
            ],
            "forecast_type": "today",
        },
    ]

    pattern = random.choice(patterns)
    return {
        "conversation": pattern["conversation"],
        "expected_function": "weather_info",
        "expected_arguments": {
            "location": loc,
            "forecast_type": pattern["forecast_type"],
        },
    }


# ============================================================
# SCHEDULE_REMINDER
# ============================================================

def generate_schedule_reminder() -> dict:
    """Generate schedule/reminder requests."""
    events = [
        ("会議", "14:00"),
        ("歯医者の予約", "10:30"),
        ("ミーティング", "15:00"),
        ("打ち合わせ", "13:00"),
        ("面接", "11:00"),
        ("飲み会", "19:00"),
        ("ジムに行く", "18:00"),
        ("薬を飲む", "21:00"),
        ("電話する", "16:00"),
        ("書類を提出する", "12:00"),
    ]

    event, time = random.choice(events)
    days = ["明日", "今日", "来週の月曜日", "金曜日", "週末"]
    day = random.choice(days)

    patterns = [
        {
            "conversation": [
                {"role": "user", "content": f"{day}の{time}に{event}があることをリマインドして。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": f"{event}の予定を登録してください。{day}の{time}です。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": f"{day}、{event}を忘れないようにしたい。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "スケジュールを追加したいです。"},
                {"role": "assistant", "content": "どのような予定ですか？"},
                {"role": "user", "content": f"{day}の{time}に{event}です。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": f"{event}のリマインダーをセットして。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": f"{day}の予定を教えて。あと、{time}に{event}を追加して。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": f"{time}に起こしてください。{event}があるので。"},
            ],
        },
    ]

    pattern = random.choice(patterns)
    return {
        "conversation": pattern["conversation"],
        "expected_function": "schedule_reminder",
        "expected_arguments": {
            "event": event,
            "datetime": f"{day} {time}",
        },
    }


# ============================================================
# SENTIMENT_LABEL
# ============================================================

def generate_sentiment_label() -> dict:
    """Generate sentiment analysis requests."""
    texts = [
        ("この製品は最高です！買って良かった。", "positive"),
        ("サービスが悪すぎる。二度と利用しない。", "negative"),
        ("まあまあでした。特に印象に残らない。", "neutral"),
        ("素晴らしい体験でした。スタッフも親切。", "positive"),
        ("期待外れでした。もっと良いと思っていた。", "negative"),
        ("普通です。可もなく不可もなく。", "neutral"),
        ("感動しました！涙が止まりませんでした。", "positive"),
        ("最悪の経験でした。時間の無駄。", "negative"),
        ("This product is amazing! Highly recommended.", "positive"),
        ("Terrible service. Never coming back.", "negative"),
    ]

    text, sentiment = random.choice(texts)

    patterns = [
        {
            "conversation": [
                {"role": "user", "content": f"「{text}」この文章の感情を分析してください。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": f"「{text}」はポジティブですか、ネガティブですか？"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": f"この文章の感情を教えて：{text}"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "レビューの感情分析をしてほしいです。"},
                {"role": "assistant", "content": "どのようなレビューですか？"},
                {"role": "user", "content": f"「{text}」です。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": f"「{text}」の感情ラベルを付けてください。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": f"このテキストのセンチメントを分析して：{text}"},
            ],
        },
    ]

    pattern = random.choice(patterns)
    return {
        "conversation": pattern["conversation"],
        "expected_function": "sentiment_label",
        "expected_arguments": {
            "text": text,
            "granularity": "basic",
        },
    }


# ============================================================
# SHOPPING_INTENT
# ============================================================

def generate_shopping_intent() -> dict:
    """Generate shopping intent detection requests."""
    products = [
        ("ノートパソコン", "research"),
        ("スマートフォン", "compare"),
        ("ヘッドフォン", "purchase"),
        ("カメラ", "research"),
        ("テレビ", "compare"),
        ("冷蔵庫", "research"),
        ("洗濯機", "purchase"),
        ("掃除機", "compare"),
        ("電子レンジ", "purchase"),
        ("イヤホン", "research"),
    ]

    product, intent = random.choice(products)

    patterns = [
        {
            "conversation": [
                {"role": "user", "content": f"{product}を買いたいのですが、おすすめはありますか？"},
            ],
            "intent": "purchase",
        },
        {
            "conversation": [
                {"role": "user", "content": f"{product}の選び方を教えてください。"},
            ],
            "intent": "research",
        },
        {
            "conversation": [
                {"role": "user", "content": f"{product}の価格を比較したいです。"},
            ],
            "intent": "compare",
        },
        {
            "conversation": [
                {"role": "user", "content": f"新しい{product}を探しています。"},
            ],
            "intent": "research",
        },
        {
            "conversation": [
                {"role": "user", "content": "買い物の相談があります。"},
                {"role": "assistant", "content": "何をお探しですか？"},
                {"role": "user", "content": f"{product}です。どれがいいか迷っています。"},
            ],
            "intent": "compare",
        },
        {
            "conversation": [
                {"role": "user", "content": f"{product}のレビューを見たいです。"},
            ],
            "intent": "research",
        },
        {
            "conversation": [
                {"role": "user", "content": f"{product}で一番コスパがいいのは？"},
            ],
            "intent": "compare",
        },
        {
            "conversation": [
                {"role": "user", "content": f"{product}を購入する予定です。"},
            ],
            "intent": "purchase",
        },
    ]

    pattern = random.choice(patterns)
    return {
        "conversation": pattern["conversation"],
        "expected_function": "shopping_intent",
        "expected_arguments": {
            "product_or_service": product,
            "intent_type": pattern.get("intent", intent),
        },
    }


# ============================================================
# MAIN
# ============================================================

GENERATORS = {
    "celebrity_info": generate_celebrity_info,
    "travel_guide": generate_travel_guide,
    "weather_info": generate_weather_info,
    "schedule_reminder": generate_schedule_reminder,
    "sentiment_label": generate_sentiment_label,
    "shopping_intent": generate_shopping_intent,
}


def add_test_ids(cases: list[dict], func_name: str, start_id: int = 1) -> list[dict]:
    """Add unique test IDs to each case."""
    for i, case in enumerate(cases):
        case["test_id"] = f"{func_name}_{start_id + i:04d}"
        case["category"] = func_name
    return cases


def generate_for_function(func_name: str, count: int, merge: bool = True) -> list[dict]:
    """Generate cases for a specific function."""
    if func_name not in GENERATORS:
        logger.error(f"Unknown function: {func_name}")
        return []

    gen_func = GENERATORS[func_name]
    cases = [gen_func() for _ in range(count)]

    if merge:
        existing_file = DATA_DIR / f"{func_name}.json"
        if existing_file.exists():
            with open(existing_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            logger.info(f"Loaded {len(existing)} existing cases for {func_name}")

            # Get max existing ID
            max_id = 0
            for case in existing:
                tid = case.get("test_id", "")
                if tid.startswith(f"{func_name}_"):
                    try:
                        num = int(tid.split("_")[-1])
                        max_id = max(max_id, num)
                    except ValueError:
                        pass

            cases = add_test_ids(cases, func_name, start_id=max_id + 1)
            cases = existing + cases
        else:
            cases = add_test_ids(cases, func_name)
    else:
        cases = add_test_ids(cases, func_name)

    return cases


def main():
    parser = argparse.ArgumentParser(description="Generate diverse function test data")
    parser.add_argument(
        "--function", "-f",
        type=str,
        default=None,
        help="Generate data for a specific function",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Generate data for all functions",
    )
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=150,
        help="Number of test cases to generate (default: 150)",
    )
    parser.add_argument(
        "--target", "-t",
        type=int,
        default=None,
        help="Target total count per function (generates difference)",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Don't merge with existing data",
    )
    args = parser.parse_args()

    if not args.function and not args.all:
        parser.error("Either --function or --all is required")

    functions = list(GENERATORS.keys()) if args.all else [args.function]

    for func_name in functions:
        # Determine count to generate
        if args.target:
            existing_file = DATA_DIR / f"{func_name}.json"
            existing_count = 0
            if existing_file.exists():
                with open(existing_file, "r") as f:
                    existing_count = len(json.load(f))
            count = max(0, args.target - existing_count)
            if count == 0:
                logger.info(f"Skipping {func_name}: already has {existing_count} >= {args.target}")
                continue
        else:
            count = args.count

        logger.info(f"Generating {count} cases for {func_name}")
        cases = generate_for_function(func_name, count, merge=not args.no_merge)

        # Save
        output_path = DATA_DIR / f"{func_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cases, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(cases)} total cases to {output_path}")

    # Update all_test_data.json
    logger.info("\nUpdating all_test_data.json...")
    all_data = []
    for f in DATA_DIR.glob("*.json"):
        if f.name == "all_test_data.json":
            continue
        with open(f, "r") as fp:
            data = json.load(fp)
            all_data.extend(data)
            print(f"  {f.name}: {len(data)} cases")

    with open(DATA_DIR / "all_test_data.json", "w") as fp:
        json.dump(all_data, fp, ensure_ascii=False, indent=2)

    print(f"\nTotal: {len(all_data)} cases saved to all_test_data.json")


if __name__ == "__main__":
    main()
