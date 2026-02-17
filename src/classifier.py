"""Stage 1 classifier for two-stage function calling.

Determines whether a function call is needed before invoking the LLM router.
This significantly improves no_function detection while maintaining function call accuracy.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ClassificationResult:
    """Result of Stage 1 classification."""
    need_function: bool
    confidence: float
    matched_function: Optional[str] = None


# Keywords that suggest NO function call is needed
NO_FUNCTION_KEYWORDS = [
    # Greetings and casual conversation
    "こんにちは", "おはよう", "こんばんは", "ありがとう", "すみません",
    "元気", "調子", "久しぶり", "よろしく", "はじめまして",
    "お疲れ様", "おやすみ", "さようなら", "またね",
    # Asking for opinions
    "どう思う", "どう思います", "意見を", "感想を",
    "思いますか", "考えますか", "でしょうか",
    # General knowledge / explanations (注: "について教えて"は人物にも使われるので除外)
    "とは何", "って何", "説明して",
    "なぜ", "どうして", "理由は", "原因は",
    "どういう意味", "違いは", "どう違う",
    # Creative / ideas / consulting
    "アイデア", "案を", "考えて", "提案", "作って",
    "書いて", "作成して", "ストーリー", "物語",
    # Math / calculation
    "計算", "足し算", "引き算", "掛け算", "割り算",
    "何倍", "パーセント",
    # Programming
    "コード", "プログラム", "バグ", "エラー", "関数",
    # Personal consultation
    "悩み", "相談", "アドバイス", "困って",
    "どうすれば", "どうしたら", "迷って",
    # Casual daily conversation
    "最近", "今日は", "趣味",
    "好き", "嫌い", "面白い", "つまらない",
    "美味しい", "きれい", "すごい",
    # Responses
    "わかりました", "了解", "なるほど", "そうですね",
]

# Strong indicators of no function (higher weight)
NO_FUNCTION_STRONG = [
    "とは何ですか", "って何ですか", "教えてください",
    "どう思いますか", "どうすればいい",
    "アドバイスをください", "相談があります",
    "理由を教えて", "違いを教えて",
]

# Keywords that suggest function call IS needed
FUNCTION_KEYWORDS = {
    "travel_guide": [
        # 基本キーワード
        "旅行", "観光", "行き方", "名所", "ホテル", "おすすめの場所",
        # 追加: 地名パターン
        "に行きたい", "の見どころ", "の名物", "観光案内", "観光情報", "観光スポット",
        # 追加: 城・寺社等
        "城について", "寺について", "神社について",
    ],
    "celebrity_info": [
        # 基本キーワード
        "有名人", "芸能人", "歌手", "俳優", "選手",
        # 追加: 人物情報パターン（「について教えて」は一般的すぎるので除外）
        "の経歴", "のプロフィール", "とは誰",
        # 追加: 職業
        "監督", "作家", "画家", "作曲家", "科学者", "アーティスト", "女優",
        # 追加: 著名人キーワード
        "の作品", "の業績", "について知りたい",
    ],
    "shopping_intent": [
        # 基本キーワード
        "買いたい", "購入", "おすすめの商品", "どこで買える", "値段",
        # 追加: 買い物パターン
        "商品を教えて", "価格を比較", "セール情報", "レビューを見せて",
        "安い", "高い", "コスパ", "お得",
    ],
    "sentiment_label": [
        # 基本キーワード
        "感情分析", "気持ち", "感情を分類", "ポジティブ", "ネガティブ",
        # 追加: 感情パターン
        "感情を分析", "感情ラベル", "センチメント", "気持ちを分析",
        "ポジティブかネガティブ", "感情を判定",
    ],
    "weather_info": [
        # 基本キーワード
        "天気", "気温", "雨", "晴れ", "予報", "明日の天気",
        # 追加: 天気パターン
        "傘は必要", "台風", "梅雨", "降水確率", "湿度",
        "曇り", "雪", "風", "嵐",
    ],
    "schedule_reminder": [
        # 基本キーワード
        "予定", "スケジュール", "リマインダー", "忘れないように", "時に通知",
        # 追加: スケジュールパターン
        "をリマインド", "会議を設定", "予約を入れ", "アラーム",
        "カレンダー", "日程", "約束",
    ],
    "translation_assist": [
        # 基本キーワード
        "翻訳", "英語にして", "日本語にして", "英訳", "和訳", "語で",
        # 追加: 言語名
        "フランス語", "韓国語", "ドイツ語", "スペイン語", "ポルトガル語",
        "イタリア語", "ロシア語", "中国語", "アラビア語",
        # 追加: 翻訳パターン
        "語に訳して", "語に変換", "語にして", "語で言うと", "語で何と言",
    ],
}


class FunctionCallClassifier:
    """Stage 1 classifier that determines if function call is needed.

    Uses keyword-based heuristics to quickly filter out conversations
    that don't require function calls, improving no_function detection.
    """

    def __init__(self, function_keywords: dict = None, no_function_keywords: list = None):
        """Initialize classifier with custom or default keywords.

        Args:
            function_keywords: Dict mapping function names to keyword lists
            no_function_keywords: List of keywords indicating no function needed
        """
        self.function_keywords = function_keywords or FUNCTION_KEYWORDS
        self.no_function_keywords = no_function_keywords or NO_FUNCTION_KEYWORDS
        self.no_function_strong = NO_FUNCTION_STRONG

    def classify(self, conversation: list[dict]) -> ClassificationResult:
        """Classify if a function call is needed.

        Args:
            conversation: List of {"role": "user"|"assistant", "content": "..."}

        Returns:
            ClassificationResult with decision and confidence
        """
        # Get last user message (most important for classification)
        last_user_msg = ""
        for msg in reversed(conversation):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break

        # Combine all user messages for context
        all_user_text = " ".join(
            msg.get("content", "") for msg in conversation if msg.get("role") == "user"
        )

        # Count no_function keywords
        no_func_score = 0
        for kw in self.no_function_keywords:
            if kw in all_user_text:
                no_func_score += 1

        # Check strong no_function indicators
        for strong in self.no_function_strong:
            pattern = strong.replace("〜", "")
            if pattern in all_user_text:
                no_func_score += 3  # Higher weight

        # Count function keywords (prioritize last user message)
        func_score = 0
        matched_function = None
        for func_name, keywords in self.function_keywords.items():
            for kw in keywords:
                if kw in last_user_msg:
                    func_score += 2  # Last message is more important
                    matched_function = func_name
                    break
                elif kw in all_user_text:
                    func_score += 1
                    matched_function = func_name
                    break

        # Decision logic
        return self._make_decision(func_score, no_func_score, matched_function)

    def _make_decision(
        self, func_score: int, no_func_score: int, matched_function: Optional[str]
    ) -> ClassificationResult:
        """Make classification decision based on scores."""

        # Strong function signal in last message
        if func_score >= 3:
            return ClassificationResult(
                need_function=True,
                confidence=min(0.6 + func_score * 0.1, 0.95),
                matched_function=matched_function,
            )

        # No function keywords and some no_function keywords
        if func_score == 0 and no_func_score >= 2:
            return ClassificationResult(
                need_function=False,
                confidence=min(0.5 + no_func_score * 0.05, 0.85),
            )

        # Function score is higher than no_function
        if func_score > no_func_score:
            return ClassificationResult(
                need_function=True,
                confidence=0.5 + (func_score - no_func_score) * 0.1,
                matched_function=matched_function,
            )

        # No_function score is significantly higher
        if no_func_score > func_score + 2:
            return ClassificationResult(
                need_function=False,
                confidence=0.5 + (no_func_score - func_score) * 0.05,
            )

        # Ambiguous case - if any function keyword, call function
        if func_score > 0:
            return ClassificationResult(
                need_function=True,
                confidence=0.5,
                matched_function=matched_function,
            )

        # Default: no function (conservative)
        if no_func_score > 0:
            return ClassificationResult(
                need_function=False,
                confidence=0.4,
            )

        # No signals either way - default to function (safer for user experience)
        return ClassificationResult(
            need_function=True,
            confidence=0.3,
        )
