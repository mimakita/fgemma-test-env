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
python -m tools.evaluate
```

Function毎のPrecision / Recall / F1、Overall Accuracy、False Positive Rateを算出します。

### Tests

```bash
pytest tests/ -v
```

## Project Structure

```
funcgemma/
├── CLAUDE.md               # Claude Code開発ガイドライン
├── README.md               # このファイル
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
│   └── evaluate.py            # 精度評価スクリプト
├── data/
│   ├── test/               # 生成されたテストデータ (JSON)
│   └── results/            # 評価結果レポート
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
