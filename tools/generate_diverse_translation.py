"""Generate diverse translation_assist test data with explicit triggers.

This script generates translation test data with clear translation intent patterns:
1. Explicit requests: 「翻訳して」「英語にして」「日本語に直して」
2. Various language pairs: ja-en, en-ja, ja-zh, ja-ko, etc.
3. Different contexts: business, casual, technical, creative
4. Various request styles: polite, casual, imperative

Usage:
    python -m tools.generate_diverse_translation --count 100
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

# Explicit translation trigger patterns
TRANSLATION_TRIGGERS = {
    "explicit_ja": [
        "を英語に翻訳して",
        "を英語にして",
        "を日本語に翻訳して",
        "を日本語にして",
        "を中国語に翻訳して",
        "を韓国語に翻訳して",
        "を翻訳してください",
        "の翻訳をお願いします",
        "を英訳して",
        "を和訳して",
        "英語で何と言いますか",
        "日本語で何と言いますか",
        "これを英語に直して",
        "これを日本語に直して",
    ],
    "explicit_en": [
        "Translate this to Japanese",
        "How do you say this in Japanese?",
        "What's this in English?",
        "Can you translate this?",
        "Please translate to Japanese",
        "I need this translated to English",
    ],
    "indirect_ja": [
        "英語でどう言えばいいですか",
        "これって英語で何て言うんだろう",
        "英語バージョンが欲しい",
        "英語に変換してほしい",
        "日本語にしてもらえますか",
        "翻訳が必要です",
        "外国語に変えて",
    ],
}

# Text samples to translate (various contexts)
TEXT_SAMPLES = {
    "business": [
        "会議の日程を変更させていただきます",
        "ご検討いただきありがとうございます",
        "契約書の確認をお願いします",
        "売上レポートを送付いたします",
        "プロジェクトの進捗状況についてご報告します",
        "来週のミーティングの件でご連絡しました",
        "予算の承認をいただけますでしょうか",
    ],
    "casual": [
        "今日の夕飯は何にする？",
        "週末どこか遊びに行かない？",
        "最近忙しくてなかなか会えないね",
        "その映画、面白かった？",
        "明日の予定は空いてる？",
        "久しぶり！元気だった？",
    ],
    "technical": [
        "このAPIのエンドポイントにアクセスできません",
        "データベースのバックアップを取ってください",
        "サーバーのレスポンス時間が遅いです",
        "このバグの再現手順を教えてください",
        "セキュリティパッチを適用する必要があります",
    ],
    "creative": [
        "桜の花が舞い散る春の午後",
        "月明かりに照らされた静かな湖",
        "希望は暗闇の中でも輝き続ける",
        "新しい冒険への第一歩を踏み出そう",
    ],
    "travel": [
        "駅への行き方を教えてください",
        "このバスは空港に行きますか？",
        "チェックインは何時からですか？",
        "おすすめのレストランはありますか？",
    ],
    "english_texts": [
        "Thank you for your consideration",
        "I look forward to hearing from you",
        "Could you please confirm the schedule?",
        "The meeting has been rescheduled",
        "I appreciate your help",
        "Let me know if you have any questions",
    ],
}

# Target languages
TARGET_LANGUAGES = {
    "english": ["英語", "English", "英訳", "en"],
    "japanese": ["日本語", "Japanese", "和訳", "ja"],
    "chinese": ["中国語", "Chinese", "zh"],
    "korean": ["韓国語", "Korean", "ko"],
    "french": ["フランス語", "French", "fr"],
    "spanish": ["スペイン語", "Spanish", "es"],
    "german": ["ドイツ語", "German", "de"],
}


def generate_explicit_request_conversation() -> dict:
    """Generate a conversation with explicit translation request."""
    trigger_type = random.choice(list(TRANSLATION_TRIGGERS.keys()))
    trigger = random.choice(TRANSLATION_TRIGGERS[trigger_type])

    # Select text to translate
    context = random.choice(list(TEXT_SAMPLES.keys()))
    text = random.choice(TEXT_SAMPLES[context])

    # Determine source/target language
    if "英語に" in trigger or "English" in trigger or "英訳" in trigger:
        target_lang = random.choice(TARGET_LANGUAGES["english"])
        source_lang = "日本語"
    elif "日本語に" in trigger or "Japanese" in trigger or "和訳" in trigger:
        target_lang = random.choice(TARGET_LANGUAGES["japanese"])
        source_lang = "English"
        text = random.choice(TEXT_SAMPLES["english_texts"])
    elif "中国語" in trigger:
        target_lang = random.choice(TARGET_LANGUAGES["chinese"])
        source_lang = "日本語"
    elif "韓国語" in trigger:
        target_lang = random.choice(TARGET_LANGUAGES["korean"])
        source_lang = "日本語"
    else:
        # Default to ja->en
        target_lang = random.choice(TARGET_LANGUAGES["english"])
        source_lang = "日本語"

    # Build conversation
    patterns = [
        # Pattern 1: Direct request
        {
            "conversation": [
                {"role": "user", "content": f"「{text}」{trigger}"},
            ],
            "text": text,
        },
        # Pattern 2: Context then request
        {
            "conversation": [
                {"role": "user", "content": "翻訳を手伝ってもらえますか？"},
                {"role": "assistant", "content": "はい、喜んでお手伝いします。何を翻訳しましょうか？"},
                {"role": "user", "content": f"「{text}」{trigger}"},
            ],
            "text": text,
        },
        # Pattern 3: Explain context then request
        {
            "conversation": [
                {"role": "user", "content": f"メールを書いているのですが、{trigger.replace('を', '')}必要があります。"},
                {"role": "assistant", "content": "どのような内容を翻訳しましょうか？"},
                {"role": "user", "content": f"「{text}」をお願いします。"},
            ],
            "text": text,
        },
        # Pattern 4: Question format
        {
            "conversation": [
                {"role": "user", "content": f"「{text}」って{target_lang}で何て言いますか？"},
            ],
            "text": text,
        },
    ]

    pattern = random.choice(patterns)

    return {
        "conversation": pattern["conversation"],
        "expected_function": "translation_assist",
        "expected_arguments": {
            "text": pattern["text"],
            "source_language": source_lang,
            "target_language": target_lang,
        },
    }


def generate_bilingual_context_conversation() -> dict:
    """Generate a conversation where bilingual context implies translation need."""
    text = random.choice(TEXT_SAMPLES["business"] + TEXT_SAMPLES["casual"])

    patterns = [
        # English assistant, Japanese user needs translation
        {
            "conversation": [
                {"role": "user", "content": "英語のメールを書きたいのですが"},
                {"role": "assistant", "content": "Sure, I can help you write an email in English. What would you like to say?"},
                {"role": "user", "content": f"「{text}」と伝えたいです。英語にしてください。"},
            ],
            "text": text,
            "target": "English",
        },
        # Document translation request
        {
            "conversation": [
                {"role": "user", "content": "海外の取引先に送る文書があります"},
                {"role": "assistant", "content": "英語に翻訳が必要ですか？"},
                {"role": "user", "content": f"はい、「{text}」を英訳してください。"},
            ],
            "text": text,
            "target": "English",
        },
        # Learning context
        {
            "conversation": [
                {"role": "user", "content": "英語の勉強をしています"},
                {"role": "assistant", "content": "頑張っていますね！何かお手伝いできることはありますか？"},
                {"role": "user", "content": f"「{text}」は英語でどう言いますか？"},
            ],
            "text": text,
            "target": "English",
        },
    ]

    pattern = random.choice(patterns)

    return {
        "conversation": pattern["conversation"],
        "expected_function": "translation_assist",
        "expected_arguments": {
            "text": pattern["text"],
            "source_language": "日本語",
            "target_language": pattern["target"],
        },
    }


def generate_specific_phrase_translation() -> dict:
    """Generate requests for translating specific phrases or idioms."""
    phrases = [
        ("一石二鳥", "英語のことわざで何と言いますか"),
        ("お疲れ様です", "英語に翻訳できますか"),
        ("よろしくお願いします", "英語でどう表現しますか"),
        ("いただきます", "英語に相当する表現を教えてください"),
        ("もったいない", "英語で何と言えばいいですか"),
        ("空気を読む", "英語で表現するとどうなりますか"),
        ("木を見て森を見ず", "英語のことわざに翻訳してください"),
    ]

    phrase, question = random.choice(phrases)

    patterns = [
        {
            "conversation": [
                {"role": "user", "content": f"「{phrase}」を{question}？"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "日本語の表現を英語にしたいのですが"},
                {"role": "assistant", "content": "どのような表現ですか？"},
                {"role": "user", "content": f"「{phrase}」です。{question}？"},
            ],
        },
    ]

    pattern = random.choice(patterns)

    return {
        "conversation": pattern["conversation"],
        "expected_function": "translation_assist",
        "expected_arguments": {
            "text": phrase,
            "source_language": "日本語",
            "target_language": "English",
        },
    }


def generate_reverse_translation() -> dict:
    """Generate English to Japanese translation requests."""
    text = random.choice(TEXT_SAMPLES["english_texts"])

    triggers = [
        f"「{text}」を日本語に翻訳してください",
        f"「{text}」を和訳してもらえますか？",
        f"「{text}」って日本語で何と言いますか？",
        f"「{text}」を日本語にしてください",
    ]

    patterns = [
        {
            "conversation": [
                {"role": "user", "content": random.choice(triggers)},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": "英語のメールをもらったのですが、意味がわかりません"},
                {"role": "assistant", "content": "どのような内容ですか？翻訳をお手伝いしましょうか？"},
                {"role": "user", "content": f"「{text}」と書いてあります。日本語に翻訳してください。"},
            ],
        },
    ]

    pattern = random.choice(patterns)

    return {
        "conversation": pattern["conversation"],
        "expected_function": "translation_assist",
        "expected_arguments": {
            "text": text,
            "source_language": "English",
            "target_language": "日本語",
        },
    }


def generate_multi_language_translation() -> dict:
    """Generate translation requests for languages other than English."""
    text = random.choice(TEXT_SAMPLES["casual"] + TEXT_SAMPLES["travel"])

    lang_pairs = [
        ("中国語", "Chinese"),
        ("韓国語", "Korean"),
        ("フランス語", "French"),
        ("スペイン語", "Spanish"),
        ("ドイツ語", "German"),
    ]

    target_ja, target_en = random.choice(lang_pairs)

    patterns = [
        {
            "conversation": [
                {"role": "user", "content": f"「{text}」を{target_ja}に翻訳してください"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": f"{target_ja}を勉強しています。「{text}」は{target_ja}で何と言いますか？"},
            ],
        },
        {
            "conversation": [
                {"role": "user", "content": f"{target_ja}圏の友達にメッセージを送りたいです"},
                {"role": "assistant", "content": f"{target_ja}への翻訳をお手伝いしましょうか？"},
                {"role": "user", "content": f"はい、「{text}」を{target_ja}にしてください。"},
            ],
        },
    ]

    pattern = random.choice(patterns)

    return {
        "conversation": pattern["conversation"],
        "expected_function": "translation_assist",
        "expected_arguments": {
            "text": text,
            "source_language": "日本語",
            "target_language": target_en,
        },
    }


def add_test_ids(cases: list[dict], start_id: int = 1) -> list[dict]:
    """Add unique test IDs to each case."""
    for i, case in enumerate(cases):
        case["test_id"] = f"translation_assist_{start_id + i:04d}"
        case["category"] = "translation_assist"
    return cases


def main():
    parser = argparse.ArgumentParser(description="Generate diverse translation_assist test data")
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=100,
        help="Number of test cases to generate (default: 100)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output filename (default: translation_assist_diverse.json)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge with existing translation_assist.json",
    )
    args = parser.parse_args()

    # Generate diverse cases
    generators = [
        (generate_explicit_request_conversation, 0.4),  # 40%
        (generate_bilingual_context_conversation, 0.2),  # 20%
        (generate_specific_phrase_translation, 0.15),   # 15%
        (generate_reverse_translation, 0.15),           # 15%
        (generate_multi_language_translation, 0.1),     # 10%
    ]

    cases = []
    for _ in range(args.count):
        # Weighted random selection of generator
        r = random.random()
        cumulative = 0
        for gen_func, weight in generators:
            cumulative += weight
            if r < cumulative:
                case = gen_func()
                cases.append(case)
                break

    logger.info(f"Generated {len(cases)} diverse translation_assist cases")

    # Optionally merge with existing data
    if args.merge:
        existing_file = DATA_DIR / "translation_assist.json"
        if existing_file.exists():
            with open(existing_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            logger.info(f"Loaded {len(existing)} existing cases")

            # Get max existing ID
            max_id = 0
            for case in existing:
                tid = case.get("test_id", "")
                if tid.startswith("translation_assist_"):
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
    output_name = args.output or ("translation_assist.json" if args.merge else "translation_assist_diverse.json")
    output_path = DATA_DIR / output_name
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved to {output_path}")

    # Show sample
    print("\n=== Sample Generated Cases ===")
    for i, case in enumerate(random.sample(cases[-args.count:], min(5, len(cases)))):
        print(f"\n[{i+1}] {case.get('test_id', 'N/A')}")
        for msg in case["conversation"]:
            print(f"  {msg['role']}: {msg['content'][:60]}...")
        print(f"  -> {case['expected_arguments']}")


if __name__ == "__main__":
    main()
