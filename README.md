# FuncGemma - Local LLM Function Calling System

[FunctionGemma](https://ollama.com/library/functiongemma) (270M) を使って、ローカルLLMとの対話履歴からリアルタイムにFunction Routingを行うシステム。

## Architecture

### 二段階Function Calling（デフォルト）

```
User Input
    │
    ▼
┌─────────────────────────────────────┐
│ Stage 1: FunctionCallClassifier    │  ← TF-IDF + LinearSVC (0.03ms)
│ (need_function / no_function 二値分類) │
└─────────────────────────────────────┘
    │                           │
    │ need_function=True        │ need_function=False
    ▼                           ▼
┌─────────────────────┐    直接応答
│ Stage 2: gemma3:4b  │    (LLM呼び出しスキップ)
│ + FunctionGemma     │
└─────────────────────┘
    │
    ▼
Function Call or None
```

- **Stage 1**: `FunctionCallClassifier` - TF-IDF + LinearSVCによる高速ML分類器
  - no_function Recall 93.7%（Accuracy 90.2%、学習時間 0.2秒）
  - 不要なLLM呼び出しを36%削減
  - `data/classifiers/stage1_model.pkl` がなければキーワードベースにフォールバック
- **Stage 2**: `FunctionGemma` - LLMベースの関数選択
  - Stage 1で関数が必要と判定された場合のみ実行

### 使用モデル

- **gemma3:4b** - 対話用メインLLM (Ollama)
- **functiongemma** - Function判定モデル 270M (Ollama)
- **qwen2.5:7b** - テストデータ生成専用 (Ollama, 単独実行)

## Registered Functions

| Function | Description | Parameters |
|----------|-------------|------------|
| `travel_guide` | 地名・旅行案内 | destination, info_type |
| `celebrity_info` | 有名人・著名人情報 | person_name, info_type |
| `shopping_intent` | 購買意図・商品/広告 | product_or_service, intent_type |
| `sentiment_label` | 感情ラベル分析 | text, granularity |
| `weather_info` | 天気情報 | location, forecast_type |
| `schedule_reminder` | スケジュール・リマインダー | action, description, datetime |
| `translation_assist` | 翻訳支援 | text, source_language, target_language |

## Requirements

- macOS (Apple Silicon)
- Python 3.9+
- Homebrew
- 8GB RAM以上

## Setup

```bash
# 全自動セットアップ (Ollama install + model pull + venv)
bash setup.sh
source .venv/bin/activate
```

### Manual Setup

```bash
# 1. Ollama install
brew install ollama
ollama serve &

# 2. Model pull
ollama pull functiongemma   # 301MB
ollama pull gemma3:4b       # 3.3GB
ollama pull qwen2.5:7b      # 4.7GB (テストデータ生成用)

# 3. Python venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Interactive Conversation

```bash
python -m src.conversation
```

対話中にFunctionGemmaが自動的にFunction判定を行い、該当するFunctionがあれば実行結果を表示します。

### Test Data Generation

```bash
# 全Function + no_function テストデータを生成 (qwen2.5:7b使用)
python -m tools.generate_test_data

# 特定のFunctionだけ生成
python -m tools.generate_test_data --function travel_guide --count 50

# no_functionデータをスキップ
python -m tools.generate_test_data --skip-no-function
```

### Classifier Testing

```bash
# Stage 1 ML分類器を学習・保存（初回またはデータ更新後に実行）
python -m tools.train_classifier
# → data/classifiers/stage1_model.pkl を生成 (学習時間 ~0.2秒)

# 分類器ベンチマーク（Keyword / TF-IDF+ML / LLM を比較）
python -m tools.benchmark_classifier

# Stage 1 Classifierの精度テスト（95件のテストケース）
python -m tools.test_conversation
```

テスト結果の例:
```
Overall Accuracy: 91/95 (95.8%)

Per-Category Results:
  greeting            : 10/10 (100.0%)
  general_question    : 10/10 (100.0%)
  creative            : 10/10 (100.0%)
  opinion             : 10/10 (100.0%)
  travel              : 10/10 (100.0%)
  weather             : 10/10 (100.0%)
  translation         : 10/10 (100.0%)
  celebrity           :  6/10 ( 60.0%)
  sentiment           :  5/ 5 (100.0%)
  schedule            :  5/ 5 (100.0%)
  shopping            :  5/ 5 (100.0%)
```

結果は `data/results/classifier_test_100.json` に保存されます。

### Evaluation

```bash
# Ollama ベースモデル評価
python -m tools.evaluate

# Fine-tuned モデル評価 (PEFT)
source .venv-ft/bin/activate
python -m tools.evaluate_peft --run-id 2 --checkpoint 800
```

Function毎のPrecision / Recall / F1、Overall Accuracy、False Positive Rateを算出します。

## Fine-tuning

FunctionGemma (270M) を LoRA でFine-tuningして精度を改善できます。

### 環境準備

```bash
# Fine-tuning 用の Python 3.12 環境を作成
python3.12 -m venv .venv-ft
source .venv-ft/bin/activate
pip install torch transformers peft accelerate datasets
```

### データ準備

```bash
# テストデータ生成（未生成の場合）
python -m tools.generate_test_data

# 学習/テストデータ分割（バランス調整あり）
python -m tools.split_data --run-id 2 --seed 42 --balance --target-count 160
```

### 学習実行

```bash
source .venv-ft/bin/activate
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # M1のメモリ制限緩和

# 学習例 (500 steps ≈ 1.7エポック)
python -m tools.finetune_peft --run-id 2 --max-steps 500

# 追加学習例 (checkpoint から再開)
python -m tools.finetune_peft --run-id 2 --max-steps 800 \
  --resume-from-checkpoint data/peft_adapters/run_2/checkpoint-500
```

### Fine-tuned モデルの使用

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# ベースモデル読み込み
base_model = AutoModelForCausalLM.from_pretrained(
    "google/functiongemma-270m-it",
    torch_dtype=torch.float32,
)
tokenizer = AutoTokenizer.from_pretrained("google/functiongemma-270m-it")

# LoRAアダプタ適用
model = PeftModel.from_pretrained(
    base_model,
    "data/peft_adapters/run_2/checkpoint-800",  # 使用するcheckpoint
)
model.eval()

# 推論
prompt = """Available tools:
- travel_guide: 地名・旅行案内
- weather_info: 天気情報

<start_of_turn>user
京都の観光スポットを教えて<end_of_turn>
<start_of_turn>model
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# => {"name": "travel_guide", "arguments": {"destination": "京都"}}
```

### 評価結果

| 構成 | Accuracy | no_function Recall | 備考 |
|------|----------|:-----------------:|------|
| Baseline (Zero-shot) | 27.9% | 低 | Ollama functiongemma |
| Run 3 (PEFT ckpt-800) | 68.3% | 0% | データ多様性向上 |
| Run 5 (PEFT ckpt-800) | 57.1% | 0% | 4,090件 balanced |
| **Run 6 (ML Stage1 + Run5)** | **93.0%** | **98.0%** | **← 最高記録** |

#### Run 6 関数別 Recall

| Function | Recall | Precision |
|----------|--------|-----------|
| travel_guide | **100.0%** | 96.1% |
| celebrity_info | **98.6%** | 100.0% |
| shopping_intent | **97.3%** | 92.3% |
| schedule_reminder | **97.3%** | 94.7% |
| weather_info | **93.2%** | 93.2% |
| sentiment_label | **83.8%** | 95.4% |
| translation_assist | 60.8% | 81.8% |
| **no_function** | **98.0%** | **100%** |

### ハイパーパラメータ

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--run-id` | required | 実験ID (データ分割と対応) |
| `--max-steps` | 500 | 最大学習ステップ数 |
| `--lora-rank` | 16 | LoRAのrank |
| `--lora-alpha` | 32 | LoRAのalpha |
| `--lr` | 3e-5 | 学習率 |
| `--batch-size` | 1 | バッチサイズ (M1 8GB制約) |
| `--max-length` | 512 | 最大シーケンス長 |

### Tests

```bash
pytest tests/ -v
```

## Project Structure

```
funcgemma/
├── CLAUDE.md               # Claude Code開発ガイドライン
├── README.md               # このファイル
├── WORKLOG.md              # 作業履歴
├── RETROSPECTIVE.md        # 振り返りと自己評価
├── requirements.txt        # Python依存パッケージ
├── setup.sh                # セットアップスクリプト
├── src/
│   ├── config.py           # モデル名・パラメータ設定
│   ├── ollama_client.py    # Ollama APIラッパー
│   ├── router.py           # FunctionGemma ルーティングロジック
│   ├── conversation.py     # 対話ループ (エントリポイント)
│   └── functions/
│       ├── registry.py     # FunctionRegistry (中央レジストリ)
│       ├── travel.py       # travel_guide
│       ├── celebrity.py    # celebrity_info
│       ├── shopping.py     # shopping_intent
│       ├── sentiment.py    # sentiment_label
│       ├── weather.py      # weather_info
│       ├── schedule.py     # schedule_reminder
│       └── translation.py  # translation_assist
├── tools/
│   ├── generate_test_data.py  # テストデータ生成 (qwen2.5:7b)
│   ├── train_classifier.py    # Stage 1 ML分類器の学習・保存
│   ├── benchmark_classifier.py # 分類器ベンチマーク (Keyword/ML/LLM比較)
│   ├── test_conversation.py   # Stage 1 精度テスト (95件)
│   ├── split_data.py          # 学習/テストデータ分割
│   ├── finetune_peft.py       # PEFT LoRA学習 (推奨)
│   ├── evaluate.py            # Ollama評価スクリプト
│   ├── evaluate_peft.py       # PEFT評価スクリプト
│   └── evaluate_run6.py       # 二段階評価 (ML Stage1 + PEFT Run5)
├── data/
│   ├── test/               # 生成されたテストデータ (JSON)
│   ├── classifiers/        # Stage 1 MLモデル
│   │   └── stage1_model.pkl  # TF-IDF + LinearSVC (919KB)
│   ├── finetune/           # 学習/テスト分割データ
│   │   └── run_{id}/       # 実験別ディレクトリ
│   ├── peft_adapters/      # Fine-tuned LoRAアダプタ
│   │   └── run_{id}/       # 実験別ディレクトリ
│   └── results/            # 評価結果レポート
├── .venv/                  # 通常実行用venv (Python 3.9)
├── .venv-ft/               # Fine-tuning用venv (Python 3.12)
└── tests/
    └── test_router.py      # ユニットテスト
```

## Memory Constraints (M1 8GB)

| 組み合わせ | 合計使用量 | 状態 |
|-----------|-----------|------|
| gemma3:4b + functiongemma | ~4GB | 同時実行OK |
| qwen2.5:7b (単独) | ~5GB | 単独実行のみ |

- `num_ctx=4096` に制限してKVキャッシュのメモリ使用量を抑制
- テストデータ生成時は他モデルをアンロードすること

## Adding a New Function

1. `src/functions/your_function.py` を作成

```python
from src.functions.registry import FunctionDefinition, FunctionRegistry

SCHEMA = {
    "type": "object",
    "properties": {
        "param1": {"type": "string", "description": "..."},
    },
    "required": ["param1"],
}

def handler(param1: str) -> dict:
    return {"function": "your_function", "param1": param1, "result": "..."}

def register(registry: FunctionRegistry):
    registry.register(FunctionDefinition(
        name="your_function",
        description="Description for FunctionGemma to understand when to route here",
        parameters=SCHEMA,
        handler=handler,
    ))
```

2. `src/functions/__init__.py` に import追加
3. `python -m tools.generate_test_data -f your_function` でテストデータ生成
4. `python -m tools.evaluate` で再評価

## License

Private project.
