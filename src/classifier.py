"""Stage 1 classifier for two-stage function calling.

Determines whether a function call is needed before invoking the LLM router.
This significantly improves no_function detection while maintaining function call accuracy.

Two backends are supported:
- ML (default): TF-IDF + LinearSVC, trained via `python -m tools.train_classifier`
  - Accuracy ~90%, latency ~0.03ms/sample
- Keyword fallback: rule-based heuristics (no training required)
  - Accuracy ~57%, latency ~0.01ms/sample
  - Used automatically if the model file is not found
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Path to the trained ML model (created by tools/train_classifier.py)
_MODEL_PATH = Path("data/classifiers/stage1_model.pkl")


@dataclass
class ClassificationResult:
    """Result of Stage 1 classification."""
    need_function: bool
    confidence: float
    matched_function: Optional[str] = None
    backend: str = "ml"  # "ml" or "keyword"


# ──────────────────────────────────────────────
# Keyword fallback (kept for when ML model is absent)
# ──────────────────────────────────────────────

# Keywords that suggest NO function call is needed
NO_FUNCTION_KEYWORDS = [
    # Greetings and casual conversation
    "こんにちは", "おはよう", "こんばんは", "ありがとう", "すみません",
    "元気", "調子", "久しぶり", "よろしく", "はじめまして",
    "お疲れ様", "おやすみ", "さようなら", "またね",
    # Asking for opinions
    "どう思う", "どう思います", "意見を", "感想を",
    "思いますか", "考えますか", "でしょうか",
    # General knowledge / explanations
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
        "旅行", "観光", "行き方", "名所", "ホテル", "おすすめの場所",
        "に行きたい", "の見どころ", "の名物", "観光案内", "観光情報", "観光スポット",
        "城について", "寺について", "神社について",
        # 口語パターン
        "で外せない", "で有名なの", "に行ったことない", "旅行するなら", "おすすめは",
        "どこ行けばいい", "何が有名",
    ],
    "celebrity_info": [
        "有名人", "芸能人", "歌手", "俳優", "選手",
        "の経歴", "のプロフィール", "とは誰",
        "監督", "作家", "画家", "作曲家", "科学者", "アーティスト", "女優",
        "の作品", "の業績", "について知りたい",
        # 口語パターン
        "ってどんな人", "って何者", "って誰", "がすごい理由", "って何した",
    ],
    "shopping_intent": [
        "買いたい", "購入", "おすすめの商品", "どこで買える", "値段",
        "商品を教えて", "価格を比較", "セール情報", "レビューを見せて",
        "安い", "高い", "コスパ", "お得",
        # 口語パターン
        "何がいい", "何がおすすめ", "どれがいい", "選び方", "比較して",
        "買おうと思ってる", "探してる", "いいの教えて",
    ],
    "sentiment_label": [
        "感情分析", "気持ち", "感情を分類", "ポジティブ", "ネガティブ",
        "感情を分析", "感情ラベル", "センチメント", "気持ちを分析",
        "ポジティブかネガティブ", "感情を判定",
    ],
    "weather_info": [
        "天気", "気温", "雨", "晴れ", "予報", "明日の天気",
        "傘は必要", "台風", "梅雨", "降水確率", "湿度",
        "曇り", "雪", "風", "嵐",
        # 口語パターン
        "雨降りそう", "雨降る？", "晴れる？", "傘いる", "天気どう",
    ],
    "schedule_reminder": [
        "予定", "スケジュール", "リマインダー", "忘れないように", "時に通知",
        "をリマインド", "会議を設定", "予約を入れ", "アラーム",
        "カレンダー", "日程", "約束",
    ],
    "translation_assist": [
        "翻訳", "英語にして", "日本語にして", "英訳", "和訳", "語で",
        "フランス語", "韓国語", "ドイツ語", "スペイン語", "ポルトガル語",
        "イタリア語", "ロシア語", "中国語", "アラビア語",
        "語に訳して", "語に変換", "語にして", "語で言うと", "語で何と言",
    ],
}


# ──────────────────────────────────────────────
# Main classifier
# ──────────────────────────────────────────────

class FunctionCallClassifier:
    """Stage 1 classifier that determines if function call is needed.

    Uses a TF-IDF + LinearSVC model (trained via tools/train_classifier.py)
    when available. Falls back to keyword-based heuristics if the model
    file is not found.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        function_keywords: Optional[dict] = None,
        no_function_keywords: Optional[list] = None,
    ):
        self.function_keywords = function_keywords or FUNCTION_KEYWORDS
        self.no_function_keywords = no_function_keywords or NO_FUNCTION_KEYWORDS
        self.no_function_strong = NO_FUNCTION_STRONG

        path = model_path or _MODEL_PATH
        self._ml_pipeline = None
        if path.exists():
            try:
                with open(path, "rb") as f:
                    self._ml_pipeline = pickle.load(f)
                logger.debug(f"Loaded ML classifier from {path}")
            except Exception as e:
                logger.warning(f"Failed to load ML classifier: {e}. Using keyword fallback.")
        else:
            logger.info(
                f"ML model not found at {path}. Using keyword fallback. "
                "Run `python -m tools.train_classifier` to build the ML model."
            )

    @property
    def backend(self) -> str:
        return "ml" if self._ml_pipeline is not None else "keyword"

    def classify(self, conversation: list[dict]) -> ClassificationResult:
        """Classify if a function call is needed.

        Args:
            conversation: List of {"role": "user"|"assistant", "content": "..."}

        Returns:
            ClassificationResult with decision and confidence
        """
        last_user_msg = ""
        for msg in reversed(conversation):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break

        if self._ml_pipeline is not None:
            return self._classify_ml(last_user_msg)
        return self._classify_keyword(last_user_msg, conversation)

    def _classify_ml(self, text: str) -> ClassificationResult:
        """Classify using the trained ML pipeline."""
        pred = self._ml_pipeline.predict([text])[0]
        # LinearSVC doesn't support predict_proba; use decision_function as proxy
        try:
            score = float(self._ml_pipeline.decision_function([text])[0])
            # Sigmoid-like mapping: score=0 → 0.5, large positive → near 1
            import math
            confidence = 1 / (1 + math.exp(-score * 0.5))
            confidence = max(0.5, min(0.99, confidence))
        except Exception:
            confidence = 0.8

        return ClassificationResult(
            need_function=bool(pred == 1),
            confidence=confidence,
            backend="ml",
        )

    def _classify_keyword(
        self, last_user_msg: str, conversation: list[dict]
    ) -> ClassificationResult:
        """Fallback: keyword-based classification."""
        all_user_text = " ".join(
            msg.get("content", "") for msg in conversation if msg.get("role") == "user"
        )

        no_func_score = sum(1 for kw in self.no_function_keywords if kw in all_user_text)
        for strong in self.no_function_strong:
            if strong.replace("〜", "") in all_user_text:
                no_func_score += 3

        func_score = 0
        matched_function = None
        for func_name, keywords in self.function_keywords.items():
            for kw in keywords:
                if kw in last_user_msg:
                    func_score += 2
                    matched_function = func_name
                    break
                elif kw in all_user_text:
                    func_score += 1
                    matched_function = func_name
                    break

        result = self._keyword_decision(func_score, no_func_score, matched_function)
        result.backend = "keyword"
        return result

    def _keyword_decision(
        self, func_score: int, no_func_score: int, matched_function: Optional[str]
    ) -> ClassificationResult:
        if func_score >= 3:
            return ClassificationResult(
                need_function=True,
                confidence=min(0.6 + func_score * 0.1, 0.95),
                matched_function=matched_function,
            )
        if func_score == 0 and no_func_score >= 2:
            return ClassificationResult(
                need_function=False,
                confidence=min(0.5 + no_func_score * 0.05, 0.85),
            )
        if func_score > no_func_score:
            return ClassificationResult(
                need_function=True,
                confidence=0.5 + (func_score - no_func_score) * 0.1,
                matched_function=matched_function,
            )
        if no_func_score > func_score + 2:
            return ClassificationResult(
                need_function=False,
                confidence=0.5 + (no_func_score - func_score) * 0.05,
            )
        if func_score > 0:
            return ClassificationResult(
                need_function=True,
                confidence=0.5,
                matched_function=matched_function,
            )
        if no_func_score > 0:
            return ClassificationResult(need_function=False, confidence=0.4)
        return ClassificationResult(need_function=True, confidence=0.3)
