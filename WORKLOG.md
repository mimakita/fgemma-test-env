# Work Log - FuncGemma

## 2025-02-04: Initial Implementation

### Phase 1: 調査・設計

- FunctionGemma の調査
  - Google が Gemma 3 270M をベースに function calling 用に特化したモデル
  - Ollama で `functiongemma` として利用可能 (301MB)
  - Ollama の tool calling API (`tools` パラメータ) をネイティブサポート
  - Zero-shot で ~58% 精度、Fine-tuning で ~85% (公式ベンチマーク)
- 対話モデルの選定
  - M1 8GB の制約上 gemma3:4b を選定 (日本語対応良好、~3.5GB)
  - functiongemma (301MB) との同時実行が可能 (~4GB合計)
- テストデータ生成モデルの選定
  - qwen2.5:7b を選定 (日本語生成品質が高い、~4.7GB)
  - メモリ制約のため単独実行のみ
- プロジェクト構成設計
  - Function Registry パターンを採用 (新規Function追加が容易)
  - Router: Ollama tools API → fallback regex parser の二段構え
  - 評価: Precision / Recall / F1 per function + Overall Accuracy

### Phase 2: 環境セットアップ

- Homebrew で Ollama v0.15.4 をインストール
- 3モデルをダウンロード:
  - `functiongemma` (300MB) - Function判定用
  - `gemma3:4b` (3.3GB) - 対話用
  - `qwen2.5:7b` (4.7GB) - テストデータ生成用
- Python 3.9 venv 作成、依存パッケージインストール (ollama, pydantic, pytest)

### Phase 3: コア実装

**作成ファイル一覧:**

| File | Description | Lines |
|------|-------------|-------|
| `setup.sh` | 全自動セットアップスクリプト | 30 |
| `requirements.txt` | Python依存パッケージ | 3 |
| `CLAUDE.md` | Claude Code開発ガイドライン | 40 |
| `src/config.py` | モデル名・パラメータ設定 | 25 |
| `src/ollama_client.py` | Ollama APIラッパー | 55 |
| `src/functions/registry.py` | FunctionRegistry + FunctionDefinition | 65 |
| `src/functions/travel.py` | travel_guide (地名・旅行案内) | 45 |
| `src/functions/celebrity.py` | celebrity_info (有名人) | 45 |
| `src/functions/shopping.py` | shopping_intent (購買意図) | 45 |
| `src/functions/sentiment.py` | sentiment_label (感情ラベル) | 45 |
| `src/functions/weather.py` | weather_info (天気情報) | 45 |
| `src/functions/schedule.py` | schedule_reminder (スケジュール) | 50 |
| `src/functions/translation.py` | translation_assist (翻訳支援) | 45 |
| `src/functions/__init__.py` | 全Function一括登録 | 20 |
| `src/router.py` | FunctionGemma ルーティングロジック | 130 |
| `src/conversation.py` | 対話ループ (メインエントリポイント) | 110 |
| `tools/generate_test_data.py` | テストデータ生成ツール | 280 |
| `tools/evaluate.py` | 精度評価スクリプト | 230 |
| `tests/test_router.py` | ユニットテスト (11 tests) | 120 |

### Phase 4: 動作確認

- FunctionGemma smoke test: tool calling API が正常に動作することを確認
  - `京都の観光スポットを教えて` → `travel_guide(destination="京都")` ✅
  - `Hello, how are you today?` → No function (正しくスキップ) ✅
- 全7 Function の routing テスト実施:
  - weather_info, sentiment_label, translation_assist: 正しくルーティング ✅
  - `こんにちは、元気ですか？`: 正しく No function ✅
  - 一部ミスルート確認 (京都観光→shopping, イチロー→travel) - zero-shot 270M の限界として想定内
- ユニットテスト: 11 tests all passing ✅

### Phase 5: テストデータ生成

- qwen2.5:7b でテストデータ生成完了 (合計 1900件)
  - 各Function: 200件
  - no_function: 500件

---

## 2025-02-10~11: LoRA Fine-tuning 実験

### 目的
FunctionGemma (270M) を LoRA Fine-tuning して、function calling 精度を改善する。

### 環境準備

```bash
# Fine-tuning 用の Python 3.12 環境を作成
python3.12 -m venv .venv-ft
source .venv-ft/bin/activate
pip install torch transformers peft accelerate datasets
```

### 試行1: MLX-LM LoRA (失敗)

Apple Silicon向けの mlx-lm を使用してLoRA学習を試みた。

**問題点:**
- 学習後のモデル出力が壊れる（`implantation)implantation...`のような無意味な文字列が出力）
- ハイパーパラメータを調整しても改善せず

**結論:** mlx-lm はこのモデルには適さないため断念。

### 試行2: Transformers + PEFT (Run 1)

HuggingFace Transformers + PEFT ライブラリに切り替え。

**設定:**
- Model: google/functiongemma-270m-it
- LoRA rank: 8, alpha: 16
- Learning rate: 2e-5
- Max steps: 200
- Training data: 1368件（不均衡: no_function 361件 vs 各function ~145件）

**問題点:**
1. 最初の学習で loss が 0.0 のまま（ラベルマスク処理のバグ）
   - 原因: tool descriptions が長すぎて max_length=512 でtruncate、応答部分が消失
   - 解決: tool descriptions を簡略化

2. MPSメモリ不足エラー
   - 解決: batch_size=1, max_length=512 に削減

**結果:**
| Metric | Baseline | Fine-tuned |
|--------|----------|------------|
| Accuracy | 28.4% | 22.6% |
| weather_info Recall | 0% | 97.5% |
| travel_guide Recall | 7.5% | 67.5% |
| celebrity_info Recall | 20% | 0% |
| schedule_reminder Recall | 0% | 0% |

**考察:**
- 一部の関数（weather_info, travel_guide）で大幅改善
- しかし全体精度は低下
- データ不均衡（no_function が多すぎ）が原因と推測

### 試行3: Balanced Data + 強化パラメータ (Run 2) - 進行中

**改善点:**
1. データバランス調整: 各カテゴリ160件に統一（合計1280件）
2. LoRA rank: 16 (8から増加)
3. LoRA alpha: 32 (16から増加)
4. Target modules: attention + MLP layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
5. Dropout: 0.1 (0.05から増加)
6. Learning rate: 3e-5 (2e-5から増加)
7. Max steps: 500 (200から増加)
8. Warmup steps: 50 (20から増加)

**進捗:**
- Checkpoint 100: loss 0.40 (順調に減少)
- 評価実行中...

### 作成・更新ファイル

| File | Description |
|------|-------------|
| `tools/split_data.py` | データ分割 + バランス調整機能追加 |
| `tools/finetune_peft.py` | PEFT LoRA学習スクリプト (推奨) |
| `tools/finetune_lora.py` | MLX LoRA学習スクリプト (非推奨) |
| `tools/evaluate_peft.py` | PEFT評価スクリプト |
| `CLAUDE.md` | ドキュメント更新 |
| `.gitignore` | Fine-tuning成果物を除外 |

### M1 8GB でのメモリ制約対応

1. `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` 環境変数でメモリ上限を緩和
2. batch_size=1, gradient_accumulation_steps=4
3. max_length=512 (デフォルト1024から削減)
4. gradient_checkpointing は無効（逆にメモリ増加したため）

### 学習時間

| 設定 | 時間 |
|------|------|
| 200 steps | ~1時間 |
| 500 steps | ~4-5時間 (推定) |

### 次のステップ

- [ ] Run 2 (500 steps) の完了を待つ
- [ ] Checkpoint 100, 200, 300, 400, 500 での精度を比較
- [ ] 最良チェックポイントを採用
- [ ] 必要に応じてさらなるハイパーパラメータ調整
