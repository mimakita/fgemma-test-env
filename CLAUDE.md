# FuncGemma - Local LLM Function Calling System

## Project Overview
FunctionGemma (270M) を使って対話履歴からFunctionルーティングを行うシステム。

## Setup
```bash
bash setup.sh        # Ollama install + model pull + venv作成
source .venv/bin/activate
```

## Architecture
- **gemma3:4b**: 対話モデル (Ollama)
- **functiongemma**: Function判定モデル 270M (Ollama)
- **qwen2.5:7b**: テストデータ生成専用 (Ollama, 単独実行)

## Memory Constraints (M1 8GB)
- gemma3:4b + functiongemma は同時実行OK (~4GB)
- qwen2.5:7b は単独実行のみ
- num_ctx=4096 に制限
- 評価実行時はfunctiongemmaを事前ロード推奨: `ollama run functiongemma "test" --keepalive 60m`
- OllamaClientはタイムアウト(30秒)とリトライ(3回)を実装済み

## Running
```bash
# 対話システム
python -m src.conversation

# テストデータ生成（初回 or 全再生成）
python -m tools.generate_test_data

# テストデータ品質修正（user終わり強制、ターン数制限、関数名リーク除去）
python -m tools.fix_test_data

# 日本語以外のテストデータを除外
python -m tools.filter_japanese_only

# 不足分の追加生成（100件未満のFunctionを補完）
python -m tools.generate_additional_data

# 評価
python -m tools.evaluate
```

## テストデータ品質要件
- 全対話は `user` メッセージで終わる（FunctionGemmaのtool_call発火条件）
- ターン数は2-4
- 関数名がテストデータ内に含まれない（リーク防止）
- 全て日本語

## Evaluation Results (Latest: Run 4)
- Run 4 (英語desc+改善データ): Accuracy 30.0%, FPR 5.6%
- Run 3 (日本語desc+改善データ): Accuracy 29.2%, FPR 1.2%
- Best Function: celebrity_info (Recall 34%, Precision 91.9% @Run4)
- 270Mモデルの限界: tool_call発火率 5-9%
- 詳細: `sample_result.md` 参照

## Adding a New Function
1. `src/functions/your_function.py` を作成 (handler + schema + register)
2. `src/functions/__init__.py` に import追加
3. テストデータ生成 → 再評価

## Functions
| Name | Description |
|------|-------------|
| travel_guide | 地名・旅行案内 |
| celebrity_info | 有名人・著名人情報 |
| shopping_intent | 購買意図・商品/広告 |
| sentiment_label | 感情ラベル |
| weather_info | 天気情報 |
| schedule_reminder | スケジュール・リマインダー |
| translation_assist | 翻訳支援 |
