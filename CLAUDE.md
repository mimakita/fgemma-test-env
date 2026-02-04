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

## Running
```bash
# 対話システム
python -m src.conversation

# テストデータ生成
python -m tools.generate_test_data

# 評価
python -m tools.evaluate
```

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
