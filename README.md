# FuncGemma - Local LLM Function Calling System

[FunctionGemma](https://ollama.com/library/functiongemma) (270M) を使って、ローカルLLMとの対話履歴からリアルタイムにFunction Routingを行うシステム。

## Architecture

```
User Input
    │
    ▼
┌──────────────┐     ┌─────────────────┐
│  gemma3:4b   │────▶│  FunctionGemma  │
│  (対話モデル)  │     │  (270M Router)  │
└──────────────┘     └────────┬────────┘
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
         ┌──────────┐  ┌──────────┐  ┌──────────┐
         │ Function │  │ Function │  │   None   │
         │    A     │  │    B     │  │(通常対話) │
         └──────────┘  └──────────┘  └──────────┘
```

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

| Model | Accuracy | Steps | Epochs | 備考 |
|-------|----------|-------|--------|------|
| Baseline (Ollama) | 28.4% | - | - | Zero-shot |
| Fine-tuned (500 steps) | **60.3%** | 500 | 1.74 | **推奨** |
| Fine-tuned (800 steps) | 60.5% | 800 | 2.78 | 微増のみ |

#### 関数別 Recall (500 steps)

| Function | Baseline | Fine-tuned | 改善 |
|----------|----------|------------|------|
| celebrity_info | 20% | **100%** | +80% |
| schedule_reminder | 0% | **95%** | +95% |
| weather_info | 0% | **92.5%** | +92.5% |
| travel_guide | 7.5% | **90%** | +82.5% |
| shopping_intent | - | 87.5% | - |
| sentiment_label | - | 65% | - |
| translation_assist | - | 42.5% | - |

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
│   ├── split_data.py          # 学習/テストデータ分割
│   ├── finetune_peft.py       # PEFT LoRA学習 (推奨)
│   ├── evaluate.py            # Ollama評価スクリプト
│   └── evaluate_peft.py       # PEFT評価スクリプト
├── data/
│   ├── test/               # 生成されたテストデータ (JSON)
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
