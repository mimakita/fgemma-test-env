"""Augment no_function training data with colloquial / topic-opening utterances.

The existing no_function data is dominated by "continuation" utterances
(e.g. "そうですね", "了解しました") within 3-turn dialogues.
Colloquial topic-openers ("暇だわ〜", "おなか減った", "人生って何？") are absent,
causing the ML classifier to misclassify them as need_function.

This script appends synthetic single-turn and 2-turn colloquial no_function
conversations to data/test/no_function.json, then rebuilds all_test_data.json.

Usage:
    python -m tools.augment_no_function [--dry-run]
"""

import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path("data/test")
NO_FUNCTION_FILE = DATA_DIR / "no_function.json"

FUNCTION_NAMES = [
    "travel_guide", "celebrity_info", "shopping_intent",
    "sentiment_label", "weather_info", "schedule_reminder", "translation_assist",
]

# ──────────────────────────────────────────────
# Synthetic colloquial no_function data
# ──────────────────────────────────────────────

# (user_text, optional_assistant_reply, optional_user_followup)
COLLOQUIAL_SAMPLES = [
    # ぼやき・感情
    ("暇だわ〜", "そうですか。何かしたいことはありますか？", "特にないんだけど"),
    ("おなか減った", "お腹が空いたんですね。何か食べましたか？", "まだ何も"),
    ("疲れた〜", "お疲れ様です。ゆっくり休んでください。", None),
    ("なんか眠い", "眠いですね。今日は何時頃起きましたか？", "6時"),
    ("やる気出ない", "そういう日もありますよね。", None),
    ("なんか憂鬱", "気分が優れないんですね。話を聞きましょうか？", "うーん"),
    ("ふあ〜眠い", "眠そうですね。昨夜はよく眠れましたか？", None),
    ("今日だるいな", "お疲れのようですね。", None),
    ("もう月曜か〜", "月曜日ですね。週の始まりです。", "憂鬱"),
    ("週末終わった", "あっという間ですよね。", "そう"),

    # 雑談・日常
    ("昨日変な夢見た", "どんな夢でしたか？", "覚えてないんだけど"),
    ("猫ってかわいいよね", "そうですね、猫は癒やされますね。", None),
    ("猫と犬どっちが好き？", "どちらもかわいいですね。あなたはどちらが好きですか？", "猫派かな"),
    ("最近ちょっと落ち込んでる", "それは大変でしたね。何かあったんですか？", "いや特には"),
    ("昨日映画見てきた", "どんな映画でしたか？", "アクション系"),
    ("今日めっちゃ寒い", "寒いですね。暖かくしてください。", None),
    ("なんかいいこと教えて", "今日も一日頑張りましょう！", "うーん"),
    ("もし1億円あったら何する？", "夢のある話ですね。あなたはどうしたいですか？", "旅行とか"),
    ("人生って難しいな", "そうですね。何かあったんですか？", "なんとなく"),
    ("なんか話して", "何について話しましょうか？", "なんでも"),

    # 意見・考察（単発）
    ("AIって結局何ができるの？", "AIはいろいろな分野で活用されています。", None),
    ("SNSって疲れない？", "そういう人も多いですね。", None),
    ("民主主義ってうまくいってると思う？", "難しい問いですね。様々な見方があります。", None),
    ("テレワークって実際どうなの", "メリットもデメリットもありますね。", None),
    ("人生って何のためにあるんだろう", "哲学的な問いですね。", None),
    ("宇宙って無限なの？", "宇宙の大きさは現代科学でも謎が多いですね。", None),
    ("AIが発達したら人間の仕事なくなると思う？", "様々な意見がありますね。", None),
    ("環境問題って解決できると思う？", "難しい課題ですね。", None),

    # 一般知識（説明依頼だがno_function）
    ("ブラックホールってどういう仕組み？", "ブラックホールは非常に重力が強い天体です。", None),
    ("光合成ってなんだっけ", "植物が光を使ってエネルギーを作る仕組みです。", None),
    ("インフレってなんで起きるの？", "物価が上昇する経済現象です。需給バランスなどが原因です。", None),
    ("ストレスが体に悪い理由は？", "ストレスはホルモンバランスや免疫系に影響します。", None),
    ("睡眠って何時間がいいの？", "一般的に7〜9時間が推奨されています。", None),
    ("なんで空は青いの？", "光の散乱現象によるものです。", None),
    ("筋肉痛ってなんで遅れてくるの？", "遅発性筋肉痛と呼ばれる現象ですね。", None),

    # 相談・悩み
    ("友達と喧嘩してしまった", "それは辛いですね。どんなことで揉めたんですか？", "些細なことで"),
    ("転職しようか迷ってる", "大きな決断ですね。何か気になることがあるんですか？", "給料とか"),
    ("彼氏と価値観が合わなくて", "それは悩ましいですね。", None),
    ("勉強のやる気が出ない", "そういう時期もありますよね。何の勉強ですか？", "英語"),
    ("ダイエットが続かない", "なかなか難しいですよね。", None),
    ("朝起きられない", "睡眠の質の問題かもしれませんね。", None),

    # クリエイティブ依頼
    ("俳句を一つ作って", "春の風　そっと頬撫で　花散る　はどうでしょう。", None),
    ("面白いなぞなぞ教えて", "では一つ。「逆さにしても読める野菜は？」", "わかった！"),
    ("10秒で笑える話して", "なぜ数学の本は悲しいのか？問題が多すぎるから。", None),
    ("ドラゴンが出てくる短い話を作って", "昔々、火を吐けないドラゴンがいました…", None),
    ("詩を書いて", "空は広く、夢は高く、今日も歩み続ける。", None),
    ("キャッチコピー考えて", "どんな商品やサービスのキャッチコピーですか？", "カフェ"),

    # 一言・超短文
    ("うーん", "何か考えていますか？", None),
    ("はあ", "どうかしましたか？", None),
    ("むずい", "何が難しいですか？", None),
    ("わからん", "何についてですか？", None),
    ("やばい", "どうしましたか？", None),
    ("えー", "何かありましたか？", None),
    ("なんか", "何かありますか？", None),
    ("ですよね", "そうですね。", None),

    # 超短口語（今回の誤分類対象）
    ("昨日すごく疲れちゃって", "大変でしたね。ゆっくり休めましたか？", None),
    ("暇だわ〜", "そうですか。何かしましょうか？", None),
    ("おなか減った", "お腹が空いたんですね。何か食べましたか？", None),
    ("もう帰りたい", "お疲れ様です。", None),
    ("眠くて死にそう", "それは辛いですね。", None),
    ("今日ついてない", "そんな日もありますよね。", None),
    ("暑すぎる", "今日は暑いですね。", None),
    ("まじつらい", "大変そうですね。何かありましたか？", None),
    ("うれしい〜", "良いことがあったんですね！", None),
    ("最高の気分", "それは良かったです！", None),

    # 相談・悩み（追加）
    ("昨日すごく疲れちゃって、どうすればいい？", "まずはゆっくり休むことが大切ですね。", None),
    ("友達と喧嘩してしまった", "それは辛いですね。仲直りできるといいですね。", None),
    ("友達に嫌なこと言われた", "それは悲しいですね。", None),
    ("転職しようか迷ってる", "大きな決断ですね。じっくり考えてみましょう。", None),
    ("仕事辞めたい", "それほど辛い状況なんですね。", None),
    ("彼女とうまくいってない", "大変ですね。何か原因はありますか？", None),
    ("親と仲悪くて", "家族関係は難しいですよね。", None),
    ("将来が不安", "将来への不安は誰にでもありますよね。", None),

    # 知識質問（曖昧だがno_function）
    ("テレワークのメリットとデメリット教えて", "通勤が不要というメリットと、孤立感というデメリットがあります。", None),
    ("チョコレートって体にいいの悪いの？", "適量であれば健康効果もあると言われています。", None),
    ("コーヒーって飲みすぎると体に悪い？", "カフェインの摂りすぎは注意が必要です。", None),
    ("運動って毎日した方がいい？", "適度な運動は健康に良いですが、休息も大切です。", None),
    ("読書って何かいいことある？", "知識が増えたり、語彙力が上がるといわれています。", None),
]


def make_conversation(user1: str, assistant1: str = None, user2: str = None) -> list[dict]:
    """Build a conversation dict."""
    messages = [{"role": "user", "content": user1}]
    if assistant1:
        messages.append({"role": "assistant", "content": assistant1})
    if user2:
        messages.append({"role": "user", "content": user2})
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print samples without saving")
    args = parser.parse_args()

    # Load existing
    existing = json.loads(NO_FUNCTION_FILE.read_text())
    print(f"Existing no_function samples: {len(existing)}")

    # Build new samples
    new_samples = []
    for i, (u1, a1, u2) in enumerate(COLLOQUIAL_SAMPLES):
        conv = make_conversation(u1, a1, u2)
        sample = {
            "conversation": conv,
            "expected_function": None,
            "expected_arguments": {},
            "test_id": f"colloquial_augment_{i:04d}",
            "category": "no_function_colloquial",
            "label": "no_function",
        }
        new_samples.append(sample)

    print(f"New colloquial samples: {len(new_samples)}")

    if args.dry_run:
        print("\nSample preview:")
        for s in random.sample(new_samples, min(5, len(new_samples))):
            last = next(m["content"] for m in reversed(s["conversation"]) if m["role"] == "user")
            print(f"  {last}")
        return

    # Append to no_function.json
    merged = existing + new_samples
    NO_FUNCTION_FILE.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
    print(f"Saved {len(merged)} samples to {NO_FUNCTION_FILE}")

    # Rebuild all_test_data.json
    all_data = list(merged)  # no_function
    for fn in FUNCTION_NAMES:
        fn_data = json.loads((DATA_DIR / f"{fn}.json").read_text())
        for r in fn_data:
            r["label"] = fn
        all_data.extend(fn_data)

    all_path = DATA_DIR / "all_test_data.json"
    all_path.write_text(json.dumps(all_data, ensure_ascii=False, indent=2))
    print(f"Rebuilt {all_path} with {len(all_data)} total samples")


if __name__ == "__main__":
    main()
