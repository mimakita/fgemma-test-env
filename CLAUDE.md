# FuncGemma - Local LLM Function Calling System

## Project Overview
FunctionGemma (270M) を使って対話履歴からFunctionルーティングを行うシステム。

## Setup
```bash
bash setup.sh        # Ollama install + model pull + venv作成
source .venv/bin/activate
```

## Architecture

### 二段階Function Calling（デフォルト）
```
User Input → Stage 1 (Classifier) → Stage 2 (FunctionGemma) → Function Call
                  ↓ no_function
              直接応答（Function呼び出しスキップ）
```

- **Stage 1**: `FunctionCallClassifier` - キーワードベースの高速分類器
  - no_function検出に特化（Recall 100%）
  - 不要なLLM呼び出しを削減
- **Stage 2**: `FunctionGemma` - LLMベースの関数選択
  - Stage 1で関数が必要と判定された場合のみ実行

### 使用モデル
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

### 対話システム
```bash
python -m src.conversation
```

### テストデータ生成
```bash
# 初期データ生成
python -m tools.generate_test_data

# データ品質修正（user終わり強制、ターン数制限、関数名リーク除去）
python -m tools.fix_test_data

# 日本語以外のテストデータを除外
python -m tools.filter_japanese_only

# 不足分の追加生成（100件未満のFunctionを補完）
python -m tools.generate_additional_data
```

### データ多様性向上ツール（Run 3以降）
```bash
# translation_assist の多様性向上（明示的トリガー追加）
python -m tools.generate_diverse_translation --count 150 --merge

# no_function の多様性向上（紛らわしいケース追加）
python -m tools.generate_diverse_no_function --count 200 --merge

# 他関数の多様性向上
python -m tools.generate_diverse_functions --all --target 370

# Irrelevance-augmented データ生成（Run 4以降）
python -m tools.generate_irrelevance_data --count 500 --ambiguous 200 --merge
```

### 評価
```bash
# Ollamaベースライン評価
python -m tools.evaluate
python -m tools.evaluate --model functiongemma-ft-run1

# PEFT Fine-tuned モデル評価
python -m tools.evaluate_peft --run-id 3 --checkpoint 800

# エラー分析
python -m tools.analyze_errors --run-id 2 --checkpoint 800 --function translation_assist
```

## LoRA Fine-tuning

### 方法1: PEFT (Transformers + PEFT) - 推奨
```bash
# 環境準備 (Python 3.12 + PEFT)
python3.12 -m venv .venv-ft
source .venv-ft/bin/activate
pip install torch transformers peft accelerate datasets

# データ分割（通常）
python -m tools.split_data --run-id 3 --seed 42

# データ分割（no_function比率制御）
python -m tools.split_data --run-id 4 --seed 42 --balance --no-function-ratio 0.5

# PEFT LoRA学習 (M1 8GBで約2-3時間)
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python -m tools.finetune_peft --run-id 4 --epochs 3 --max-steps 800

# 評価
python -m tools.evaluate_peft --run-id 4 --checkpoint 800
```

### 方法2: MLX-LM (非推奨)
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

### Run 3: データ多様性向上 (checkpoint-800)
| Metric | Value |
|--------|-------|
| Accuracy | **68.3%** |
| celebrity_info Recall | 98.6% |
| travel_guide Recall | 98.6% |
| schedule_reminder Recall | 98.6% |
| shopping_intent Recall | 97.3% |
| weather_info Recall | 93.2% |
| sentiment_label Recall | 78.4% |
| translation_assist Recall | 51.4% |
| no_function Recall | 0% (課題) |

### Fine-tuning効果
- ベースラインから **+40.4%** の精度向上
- データ多様性向上により translation_assist が +13.9% 改善
- no_function の認識が継続課題（全て関数呼び出しと判定）

### no_function 認識改善への取り組み (Run 4)
- **Irrelevance-Augmented Dataset**: 関数除外型ネガティブサンプル追加
- **no_function 比率50%**: 学習データの半分をno_functionに
- 参考文献: Decision Token (arxiv:2412.01130v2), Hammer (arxiv:2410.04587v2)

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
│   ├── router.py          # Function Router（二段階判定統合）
│   ├── classifier.py      # Stage 1 Classifier（キーワードベース）
│   ├── config.py          # 設定
│   └── functions/         # Function定義
├── tools/
│   ├── utils/                        # 共通ユーティリティ
│   │   ├── common.py                 # パス定義、JSON処理、プロンプト生成
│   │   └── __init__.py
│   ├── datagen/                      # データ生成ツール
│   ├── training/                     # 学習ツール
│   ├── eval/                         # 評価ツール
│   ├── generate_test_data.py         # テストデータ生成
│   ├── generate_additional_data.py   # 追加データ生成
│   ├── generate_diverse_translation.py  # translation_assist多様性向上
│   ├── generate_diverse_no_function.py  # no_function多様性向上
│   ├── generate_diverse_functions.py    # 他関数多様性向上
│   ├── generate_irrelevance_data.py     # Irrelevance-augmented データ
│   ├── fix_test_data.py              # データ品質修正
│   ├── filter_japanese_only.py       # 日本語フィルター
│   ├── split_data.py                 # 学習/テスト分割（--no-function-ratio対応）
│   ├── finetune_peft.py              # PEFT LoRA学習 (推奨)
│   ├── finetune_lora.py              # MLX LoRA学習 (非推奨)
│   ├── evaluate.py                   # Ollama評価
│   ├── evaluate_peft.py              # PEFT評価
│   └── analyze_errors.py             # エラー分析
├── data/
│   ├── test/                         # 元テストデータ
│   ├── finetune/run_{id}/            # 分割済みデータ
│   ├── peft_adapters/run_{id}/       # PEFTアダプタ
│   └── results/                      # 評価結果JSON
├── sample_result.md                  # 評価結果レポート
└── .venv-ft/                         # Fine-tuning用venv
```

## 実験履歴

| Run | データ件数 | Accuracy | 特徴 |
|-----|----------|----------|------|
| Baseline | 950 | 27.9% | Zero-shot |
| Run 2 | 1,900 | 60.3% | PEFT Fine-tuning |
| Run 3 | 3,340 | **68.3%** | データ多様性向上（ベスト）|
| Run 4 | 4,790 | 42.7% | no_function 50%比率（失敗）|
| Run 5 | 4,090 | 57.1% | balanced, 12.5% no_function |

### Run 4 失敗の原因
- no_function 50%にしたことで各関数の学習サンプルが7%に減少
- function_excluded/ambiguous_query データが混乱を引き起こした
- Run 5ではno_functionをロールバック（2,200→1,500件）して12.5%に

### 重要な発見
- **no_function認識は単純なデータ増強では解決しない** (全Runで Recall 0%)
- **Decision Token** などのアーキテクチャ変更が必要
- **データバランスが重要**: no_function比率が高すぎると全体精度が低下

### 二段階判定（本番統合済み）

| メトリクス | 単独モデル | 二段階判定 | 改善 |
|-----------|-----------|-----------|------|
| Accuracy | 57.1% | **70.8%** | +13.7% |
| no_function Recall | 0% | **100%** | +100% |

**結論**: 二段階判定でno_function認識問題を解決し、本番コードに統合

#### 実装詳細
- `src/classifier.py`: Stage 1 分類器（キーワードベース）
- `src/router.py`: `FunctionRouter.use_two_stage=True` がデフォルト
- プロトタイプ検証: `python -m tools.prototype_two_stage --run-id 5`

詳細は `sample_result.md` を参照。
