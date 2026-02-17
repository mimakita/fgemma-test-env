"""Generate diverse no_function test data.

This script generates conversations that should NOT trigger any function.
Includes "confusing" cases that might look like function calls but aren't.

Categories:
1. Pure chitchat/greetings
2. General knowledge questions
3. Opinion/advice requests
4. Math/logic problems
5. Creative writing requests
6. Personal conversations
7. Technical discussions (non-actionable)
8. Philosophical discussions
9. CONFUSING: Translation-like but not (key for reducing FP)
10. CONFUSING: Weather mentions but not requests
11. CONFUSING: Travel mentions but not requests
12. CONFUSING: Celebrity mentions but not lookups
13. CONFUSING: Schedule mentions but not reminders
14. CONFUSING: Shopping mentions but not intent

Usage:
    python -m tools.generate_diverse_no_function --count 200
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
# CONFUSING CASES - Look like function calls but aren't
# ============================================================

def generate_translation_like_not_translation() -> dict:
    """Generate conversations that mention translation/languages but don't request it."""
    patterns = [
        # Talking ABOUT translation, not requesting it
        {
            "conversation": [
                {"role": "user", "content": "翻訳の仕事って難しいですか？"},
                {"role": "assistant", "content": "専門的なスキルが必要ですね。言語能力だけでなく文化理解も大切です。"},
                {"role": "user", "content": "そうですね、AIの翻訳精度も上がってきましたよね。"},
            ],
        },
        # Learning languages (not translating)
        {
            "conversation": [
                {"role": "user", "content": "英語を勉強しています。"},
                {"role": "assistant", "content": "素晴らしいですね！どのくらい勉強されていますか？"},
                {"role": "user", "content": "3年くらいです。まだまだですけど。"},
            ],
        },
        # Bilingual conversation without translation request
        {
            "conversation": [
                {"role": "user", "content": "今日は良い天気ですね。"},
                {"role": "assistant", "content": "Yes, it's a beautiful day!"},
                {"role": "user", "content": "散歩に行こうかな。"},
            ],
        },
        # Mentioning foreign words without asking for translation
        {
            "conversation": [
                {"role": "user", "content": "「サステナビリティ」って最近よく聞きますね。"},
                {"role": "assistant", "content": "環境問題への意識が高まっていますからね。"},
                {"role": "user", "content": "企業も取り組みを強化していますよね。"},
            ],
        },
        # Past translation experience
        {
            "conversation": [
                {"role": "user", "content": "昨日、友達に手紙を翻訳してもらったんです。"},
                {"role": "assistant", "content": "それは助かりましたね。何語の手紙でしたか？"},
                {"role": "user", "content": "フランス語でした。友達は留学経験があるので。"},
            ],
        },
        # Discussing translation tools
        {
            "conversation": [
                {"role": "user", "content": "Google翻訳って便利ですよね。"},
                {"role": "assistant", "content": "そうですね、精度も年々向上しています。"},
                {"role": "user", "content": "でも微妙なニュアンスはまだ難しいですよね。"},
            ],
        },
        # Language preference (not translation)
        {
            "conversation": [
                {"role": "user", "content": "英語と日本語、どちらが好きですか？"},
                {"role": "assistant", "content": "どちらも魅力的な言語ですね。あなたはどちらが好きですか？"},
                {"role": "user", "content": "日本語の方が表現が豊かだと思います。"},
            ],
        },
        # Foreign language movies/media
        {
            "conversation": [
                {"role": "user", "content": "字幕なしで英語の映画を見られるようになりたい。"},
                {"role": "assistant", "content": "リスニング力を鍛えるのに良い目標ですね。"},
                {"role": "user", "content": "何かおすすめの練習方法はありますか？"},
            ],
        },
    ]
    return random.choice(patterns)


def generate_weather_like_not_weather() -> dict:
    """Generate conversations that mention weather but don't request forecast."""
    patterns = [
        # Past weather
        {
            "conversation": [
                {"role": "user", "content": "昨日は雨がすごかったですね。"},
                {"role": "assistant", "content": "本当に大変でしたね。傘を持っていましたか？"},
                {"role": "user", "content": "忘れて濡れちゃいました。"},
            ],
        },
        # Weather as small talk
        {
            "conversation": [
                {"role": "user", "content": "今日は暖かいですね。"},
                {"role": "assistant", "content": "春らしい陽気ですね。"},
                {"role": "user", "content": "桜もそろそろ咲きそうです。"},
            ],
        },
        # Discussing climate
        {
            "conversation": [
                {"role": "user", "content": "最近、気候変動について考えることが多いです。"},
                {"role": "assistant", "content": "地球温暖化は深刻な問題ですね。"},
                {"role": "user", "content": "私たちにできることは何でしょうか。"},
            ],
        },
        # Weather preference
        {
            "conversation": [
                {"role": "user", "content": "晴れの日と雨の日、どちらが好きですか？"},
                {"role": "assistant", "content": "晴れの日は気持ちいいですが、雨も落ち着きますね。"},
                {"role": "user", "content": "私は晴れが好きです。洗濯物が乾くので。"},
            ],
        },
        # Weather metaphor
        {
            "conversation": [
                {"role": "user", "content": "人生は天気みたいなものですね。"},
                {"role": "assistant", "content": "晴れの日もあれば雨の日もありますね。"},
                {"role": "user", "content": "その通りです。いつも晴れとは限らない。"},
            ],
        },
    ]
    return random.choice(patterns)


def generate_travel_like_not_travel() -> dict:
    """Generate conversations that mention travel but don't request guide info."""
    patterns = [
        # Past travel
        {
            "conversation": [
                {"role": "user", "content": "去年、京都に行きました。"},
                {"role": "assistant", "content": "いいですね！どうでしたか？"},
                {"role": "user", "content": "紅葉がとても綺麗でした。"},
            ],
        },
        # Travel dreams
        {
            "conversation": [
                {"role": "user", "content": "いつかヨーロッパに行ってみたいです。"},
                {"role": "assistant", "content": "素敵な夢ですね。どの国に興味がありますか？"},
                {"role": "user", "content": "フランスとイタリアかな。"},
            ],
        },
        # Travel industry discussion
        {
            "conversation": [
                {"role": "user", "content": "コロナで旅行業界は大変でしたね。"},
                {"role": "assistant", "content": "本当に厳しい時期でしたね。"},
                {"role": "user", "content": "最近は回復してきているようですが。"},
            ],
        },
        # Local area (not tourist)
        {
            "conversation": [
                {"role": "user", "content": "近所に新しいカフェができました。"},
                {"role": "assistant", "content": "行ってみましたか？"},
                {"role": "user", "content": "まだですが、今度行ってみようと思います。"},
            ],
        },
    ]
    return random.choice(patterns)


def generate_celebrity_like_not_celebrity() -> dict:
    """Generate conversations that mention celebrities but don't request info lookup."""
    patterns = [
        # Opinion about celebrity
        {
            "conversation": [
                {"role": "user", "content": "最近の俳優さんは演技が上手いですよね。"},
                {"role": "assistant", "content": "そうですね、レベルが上がっていると思います。"},
                {"role": "user", "content": "昔の映画も良いですけどね。"},
            ],
        },
        # Celebrity news as conversation
        {
            "conversation": [
                {"role": "user", "content": "芸能ニュースって見ますか？"},
                {"role": "assistant", "content": "たまに見ることがあります。"},
                {"role": "user", "content": "私はあまり興味がないんですよね。"},
            ],
        },
        # Fan discussion
        {
            "conversation": [
                {"role": "user", "content": "好きなアーティストのライブに行きたいです。"},
                {"role": "assistant", "content": "いいですね！誰のライブですか？"},
                {"role": "user", "content": "秘密です。チケット取れるか分からないので。"},
            ],
        },
    ]
    return random.choice(patterns)


def generate_schedule_like_not_schedule() -> dict:
    """Generate conversations that mention schedules but don't request reminders."""
    patterns = [
        # Talking about busyness
        {
            "conversation": [
                {"role": "user", "content": "今週は予定がいっぱいです。"},
                {"role": "assistant", "content": "お忙しそうですね。体調管理には気をつけてください。"},
                {"role": "user", "content": "ありがとうございます。頑張ります。"},
            ],
        },
        # Past schedule
        {
            "conversation": [
                {"role": "user", "content": "昨日の会議は長かったです。"},
                {"role": "assistant", "content": "大変でしたね。何時間くらいでしたか？"},
                {"role": "user", "content": "3時間くらいでした。"},
            ],
        },
        # Schedule flexibility
        {
            "conversation": [
                {"role": "user", "content": "フリーランスは時間の自由があっていいですね。"},
                {"role": "assistant", "content": "自己管理が大切になりますね。"},
                {"role": "user", "content": "確かに、だらけてしまうこともあります。"},
            ],
        },
    ]
    return random.choice(patterns)


def generate_shopping_like_not_shopping() -> dict:
    """Generate conversations that mention shopping but don't express purchase intent."""
    patterns = [
        # Past purchase
        {
            "conversation": [
                {"role": "user", "content": "先週、新しい靴を買いました。"},
                {"role": "assistant", "content": "いいですね！どんな靴ですか？"},
                {"role": "user", "content": "スニーカーです。歩きやすいです。"},
            ],
        },
        # Window shopping
        {
            "conversation": [
                {"role": "user", "content": "デパートをぶらぶら見るのが好きです。"},
                {"role": "assistant", "content": "ウィンドウショッピングは楽しいですよね。"},
                {"role": "user", "content": "買わなくても見るだけで満足します。"},
            ],
        },
        # Discussing consumerism
        {
            "conversation": [
                {"role": "user", "content": "最近、断捨離を始めました。"},
                {"role": "assistant", "content": "いいですね。物を減らすのは気持ちいいですよね。"},
                {"role": "user", "content": "無駄な買い物も減りました。"},
            ],
        },
        # Gift received
        {
            "conversation": [
                {"role": "user", "content": "誕生日にプレゼントをもらいました。"},
                {"role": "assistant", "content": "おめでとうございます！何をもらったんですか？"},
                {"role": "user", "content": "時計です。とても嬉しかったです。"},
            ],
        },
    ]
    return random.choice(patterns)


def generate_sentiment_like_not_sentiment() -> dict:
    """Generate emotional conversations that don't need sentiment analysis."""
    patterns = [
        # Expressing feelings (not asking for analysis)
        {
            "conversation": [
                {"role": "user", "content": "今日は気分がいいです。"},
                {"role": "assistant", "content": "それは良かったですね！何かいいことがありましたか？"},
                {"role": "user", "content": "特にないですけど、天気がいいからかな。"},
            ],
        },
        # Discussing emotions
        {
            "conversation": [
                {"role": "user", "content": "感情をコントロールするのって難しいですね。"},
                {"role": "assistant", "content": "確かに、イライラした時は特に大変ですよね。"},
                {"role": "user", "content": "深呼吸すると少し落ち着きます。"},
            ],
        },
    ]
    return random.choice(patterns)


# ============================================================
# PURE NO-FUNCTION CASES
# ============================================================

def generate_greeting() -> dict:
    """Generate greeting conversations."""
    patterns = [
        {
            "conversation": [
                {"role": "user", "content": "こんにちは！"},
                {"role": "assistant", "content": "こんにちは！今日はどのようなご用件ですか？"},
                {"role": "user", "content": "特にないです。ただ挨拶しに来ました。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "おはようございます。"},
                {"role": "assistant", "content": "おはようございます！良い朝ですね。"},
                {"role": "user", "content": "はい、気持ちいいですね。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "お疲れ様です。"},
                {"role": "assistant", "content": "お疲れ様です！一日どうでしたか？"},
                {"role": "user", "content": "まあまあでした。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "やあ、元気？"},
                {"role": "assistant", "content": "元気ですよ！あなたは？"},
                {"role": "user", "content": "僕も元気です。"},
            ],
        },
    ]
    return random.choice(patterns)


def generate_general_knowledge() -> dict:
    """Generate general knowledge questions."""
    patterns = [
        {
            "conversation": [
                {"role": "user", "content": "地球から月までの距離はどのくらいですか？"},
                {"role": "assistant", "content": "約38万キロメートルです。"},
                {"role": "user", "content": "思ったより遠いですね。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "日本で一番高い山は何ですか？"},
                {"role": "assistant", "content": "富士山で、標高は約3,776メートルです。"},
                {"role": "user", "content": "一度登ってみたいです。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "水の沸点は何度ですか？"},
                {"role": "assistant", "content": "1気圧では100度です。"},
                {"role": "user", "content": "高地だと低くなるんですよね。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "第二次世界大戦はいつ終わりましたか？"},
                {"role": "assistant", "content": "1945年に終結しました。"},
                {"role": "user", "content": "もう80年近く前なんですね。"},
            ],
        },
    ]
    return random.choice(patterns)


def generate_opinion_request() -> dict:
    """Generate opinion/advice conversations."""
    patterns = [
        {
            "conversation": [
                {"role": "user", "content": "犬と猫、どっちがペットとしていいと思いますか？"},
                {"role": "assistant", "content": "どちらも良いですが、生活スタイルによりますね。"},
                {"role": "user", "content": "一人暮らしなら猫の方がいいですかね。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "朝型と夜型、どちらが健康にいいと思いますか？"},
                {"role": "assistant", "content": "一般的には朝型が良いとされていますね。"},
                {"role": "user", "content": "夜型を直すのは難しいですよね。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "読書は紙の本と電子書籍、どちらがいいですか？"},
                {"role": "assistant", "content": "好みによりますが、それぞれ長所がありますね。"},
                {"role": "user", "content": "私は紙の本の方が好きです。"},
            ],
        },
    ]
    return random.choice(patterns)


def generate_math_logic() -> dict:
    """Generate math/logic problem conversations."""
    patterns = [
        {
            "conversation": [
                {"role": "user", "content": "123 × 456 はいくつですか？"},
                {"role": "assistant", "content": "56,088です。"},
                {"role": "user", "content": "暗算で計算できますか？"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "1から100までの合計は？"},
                {"role": "assistant", "content": "5,050です。"},
                {"role": "user", "content": "公式があるんですよね。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "√2 の値は大体いくつですか？"},
                {"role": "assistant", "content": "約1.414です。"},
                {"role": "user", "content": "無理数なんですよね。"},
            ],
        },
    ]
    return random.choice(patterns)


def generate_creative() -> dict:
    """Generate creative/hypothetical conversations."""
    patterns = [
        {
            "conversation": [
                {"role": "user", "content": "もしタイムマシンがあったらどの時代に行きたいですか？"},
                {"role": "assistant", "content": "恐竜の時代を見てみたいですね。"},
                {"role": "user", "content": "危険そうですけど、面白そうですね。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "宇宙人は存在すると思いますか？"},
                {"role": "assistant", "content": "宇宙の広さを考えると、いる可能性は高いですね。"},
                {"role": "user", "content": "いつか会えるといいですね。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "100年後の世界はどうなっていると思いますか？"},
                {"role": "assistant", "content": "技術がさらに発展しているでしょうね。"},
                {"role": "user", "content": "AIも進化しているでしょうね。"},
            ],
        },
    ]
    return random.choice(patterns)


def generate_personal() -> dict:
    """Generate personal conversation topics."""
    patterns = [
        {
            "conversation": [
                {"role": "user", "content": "趣味は何ですか？"},
                {"role": "assistant", "content": "私は人工知能なので趣味はありませんが、あなたは？"},
                {"role": "user", "content": "読書とゲームが好きです。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "最近、料理を始めました。"},
                {"role": "assistant", "content": "素晴らしいですね！何を作りましたか？"},
                {"role": "user", "content": "カレーを作りました。意外と簡単でした。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "子供の頃の夢は何でしたか？"},
                {"role": "assistant", "content": "どんな夢を持っていましたか？"},
                {"role": "user", "content": "宇宙飛行士になりたかったです。"},
            ],
        },
    ]
    return random.choice(patterns)


def generate_philosophical() -> dict:
    """Generate philosophical discussions."""
    patterns = [
        {
            "conversation": [
                {"role": "user", "content": "幸せって何だと思いますか？"},
                {"role": "assistant", "content": "人それぞれ定義が違いますね。"},
                {"role": "user", "content": "確かに、人によって違いますよね。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "人生の意味って何だと思いますか？"},
                {"role": "assistant", "content": "難しい質問ですね。自分で見つけるものかもしれません。"},
                {"role": "user", "content": "深いですね。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "AIは意識を持てると思いますか？"},
                {"role": "assistant", "content": "哲学的に難しい問題ですね。"},
                {"role": "user", "content": "いつか答えが出るといいですね。"},
            ],
        },
    ]
    return random.choice(patterns)


def generate_tech_discussion() -> dict:
    """Generate technical discussions (non-actionable)."""
    patterns = [
        {
            "conversation": [
                {"role": "user", "content": "プログラミング言語は何が好きですか？"},
                {"role": "assistant", "content": "Pythonは読みやすくて人気がありますね。"},
                {"role": "user", "content": "私もPythonを勉強しています。"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "スマートフォンの進化はすごいですね。"},
                {"role": "assistant", "content": "10年前と比べると全然違いますね。"},
                {"role": "user", "content": "次は何が来ると思いますか？"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "インターネットがない時代って想像できますか？"},
                {"role": "assistant", "content": "今では生活に欠かせないものになりましたね。"},
                {"role": "user", "content": "本当にそうですね。"},
            ],
        },
    ]
    return random.choice(patterns)


def add_test_ids(cases: list[dict], start_id: int = 1) -> list[dict]:
    """Add unique test IDs to each case."""
    for i, case in enumerate(cases):
        case["test_id"] = f"no_function_{start_id + i:04d}"
        case["category"] = "no_function"
        case["expected_function"] = None
        case["expected_arguments"] = None
    return cases


def main():
    parser = argparse.ArgumentParser(description="Generate diverse no_function test data")
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=200,
        help="Number of test cases to generate (default: 200)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output filename (default: no_function_diverse.json)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge with existing no_function.json",
    )
    args = parser.parse_args()

    # Generator functions with weights
    # Higher weight for confusing cases to help model learn boundaries
    generators = [
        # Confusing cases (50% - crucial for reducing FP)
        (generate_translation_like_not_translation, 0.12),
        (generate_weather_like_not_weather, 0.08),
        (generate_travel_like_not_travel, 0.08),
        (generate_celebrity_like_not_celebrity, 0.06),
        (generate_schedule_like_not_schedule, 0.06),
        (generate_shopping_like_not_shopping, 0.06),
        (generate_sentiment_like_not_sentiment, 0.04),
        # Pure no-function cases (50%)
        (generate_greeting, 0.08),
        (generate_general_knowledge, 0.10),
        (generate_opinion_request, 0.08),
        (generate_math_logic, 0.06),
        (generate_creative, 0.06),
        (generate_personal, 0.06),
        (generate_philosophical, 0.04),
        (generate_tech_discussion, 0.02),
    ]

    cases = []
    for _ in range(args.count):
        # Weighted random selection
        r = random.random()
        cumulative = 0
        for gen_func, weight in generators:
            cumulative += weight
            if r < cumulative:
                case = gen_func()
                cases.append(case)
                break

    logger.info(f"Generated {len(cases)} diverse no_function cases")

    # Optionally merge with existing data
    if args.merge:
        existing_file = DATA_DIR / "no_function.json"
        if existing_file.exists():
            with open(existing_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            logger.info(f"Loaded {len(existing)} existing cases")

            # Get max existing ID
            max_id = 0
            for case in existing:
                tid = case.get("test_id", "")
                if tid.startswith("no_function_"):
                    try:
                        num = int(tid.split("_")[-1])
                        max_id = max(max_id, num)
                    except ValueError:
                        pass

            cases = add_test_ids(cases, start_id=max_id + 1)
            cases = existing + cases
            logger.info(f"Total after merge: {len(cases)} cases")
        else:
            cases = add_test_ids(cases)
    else:
        cases = add_test_ids(cases)

    # Save
    output_name = args.output or ("no_function.json" if args.merge else "no_function_diverse.json")
    output_path = DATA_DIR / output_name
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved to {output_path}")

    # Show distribution
    print("\n=== Generated Case Distribution ===")
    categories = {}
    for case in cases[-args.count:]:
        # Identify category by first user message
        first_msg = ""
        for msg in case.get("conversation", []):
            if msg.get("role") == "user":
                first_msg = msg.get("content", "")[:30]
                break
        # Simple categorization
        if "翻訳" in first_msg or "英語" in first_msg or "言語" in first_msg:
            cat = "translation-like"
        elif "天気" in first_msg or "雨" in first_msg or "晴" in first_msg:
            cat = "weather-like"
        elif "旅" in first_msg or "行き" in first_msg:
            cat = "travel-like"
        elif "こんにち" in first_msg or "おはよう" in first_msg:
            cat = "greeting"
        else:
            cat = "other"
        categories[cat] = categories.get(cat, 0) + 1

    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Show samples
    print("\n=== Sample Generated Cases ===")
    for i, case in enumerate(random.sample(cases[-args.count:], min(5, len(cases)))):
        print(f"\n[{i+1}] {case.get('test_id', 'N/A')}")
        for msg in case["conversation"]:
            print(f"  {msg['role']}: {msg['content'][:50]}...")


if __name__ == "__main__":
    main()
