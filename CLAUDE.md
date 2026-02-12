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
python -m tools.evaluate --model functiongemma-ft-run1  # Fine-tunedモデル評価
```

## LoRA Fine-tuning

### 方法1: PEFT (Transformers + PEFT) - 推奨
```bash
# 環境準備 (Python 3.12 + PEFT)
python3.12 -m venv .venv-ft
source .venv-ft/bin/activate
pip install torch transformers peft accelerate datasets

# データ分割
python -m tools.split_data --run-id 1 --seed 42

# PEFT LoRA学習 (M1 8GBで約1時間)
python -m tools.finetune_peft --run-id 1 --epochs 3

# 評価 (Transformersで直接)
python -m tools.evaluate_peft --run-id 2 --checkpoint 500
```

### 方法2: MLX-LM (非推奨 - モデル出力が壊れる問題あり)
```bash
# mlx-lmでの学習は270Mモデルで出力が壊れる問題があるため非推奨
python -m tools.finetune_lora --run-id 1  # 使用しないこと
```

## テストデータ品質要件
- 全対話は `user` メッセージで終わる（FunctionGemmaのtool_call発火条件）
- ターン数は2-4
- 関数名がテストデータ内に含まれない（リーク防止）
- 全て日本語

## Evaluation Results

### ベースライン (Ollama functiongemma, Zero-shot)
| Metric | Value |
|--------|-------|
| Accuracy | 27.9% |
| FPR | 4.8% |
| celebrity_info Recall | 17.5% |
| travel_guide Recall | 4.0% |
| 他のfunction | 0-2.5% |

### PEFT Fine-tuned (500 steps, 推奨)
| Metric | Value |
|--------|-------|
| Accuracy | **60.3%** |
| celebrity_info Recall | 100% |
| schedule_reminder Recall | 95.0% |
| weather_info Recall | 92.5% |
| travel_guide Recall | 90.0% |
| shopping_intent Recall | 87.5% |
| sentiment_label Recall | 65.0% |
| translation_assist Recall | 42.5% |
| no_function Recall | 0% (課題) |

### Fine-tuning効果
- ベースラインから **+32.4%** の精度向上
- 500 steps (約1.7エポック) で十分な精度に到達
- 800 steps でも +0.2% のみ（収束済み）
- no_function の認識が課題（全て関数呼び出しと判定）

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

## ディレクトリ構造
```
funcgemma/
├── src/
│   ├── conversation.py    # 対話システム
│   ├── router.py          # Function Router
│   ├── config.py          # 設定
│   └── functions/         # Function定義
├── tools/
│   ├── generate_test_data.py      # テストデータ生成
│   ├── generate_additional_data.py # 追加データ生成
│   ├── fix_test_data.py           # データ品質修正
│   ├── filter_japanese_only.py    # 日本語フィルター
│   ├── split_data.py              # 学習/テスト分割
│   ├── finetune_peft.py           # PEFT LoRA学習 (推奨)
│   ├── finetune_lora.py           # MLX LoRA学習 (非推奨)
│   ├── evaluate.py                # Ollama評価
│   └── evaluate_peft.py           # PEFT評価
├── data/
│   ├── test/                      # 元テストデータ
│   ├── finetune/run_{id}/         # 分割済みデータ
│   └── peft_adapters/run_{id}/    # PEFTアダプタ
└── .venv-ft/                      # Fine-tuning用venv
```
