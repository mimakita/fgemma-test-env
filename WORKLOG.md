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

### Phase 5: テストデータ生成 (進行中)

- qwen2.5:7b でテストデータ生成を開始
- 生成完了:
  - `travel_guide.json`: 100件 ✅
  - `celebrity_info.json`: 100件 ✅
  - `shopping_intent.json`: 100件 ✅
- 残り:
  - `sentiment_label`: 生成中
  - `weather_info`: 未着手
  - `schedule_reminder`: 未着手
  - `translation_assist`: 未着手
  - `no_function`: 250件 未着手
- M1 8GB での qwen2.5:7b は 1バッチ(10件)あたり2-4分程度

### 設計上の判断事項

1. **Router の二段構え**: Ollama の構造化 `tool_calls` を優先し、失敗時は raw output の regex パースにフォールバック
2. **num_ctx=4096**: 8GB RAM でのKVキャッシュ節約のため 32K ではなく 4096 に制限
3. **keep_alive 戦略**: gemma3:4b は 10m、functiongemma は 5m (小さいので再ロードが速い)
4. **テストデータ生成**: バッチサイズ 10、最大3回リトライ、JSON バリデーション + 重複排除
5. **提案追加Function**: weather_info, schedule_reminder, translation_assist の3つを追加 (合計7 Function)

### 既知の課題

- FunctionGemma zero-shot 精度は公式ベンチマークで ~58%、Fine-tuning なしでは限界がある
- 日本語入力での一部ミスルート (Function description を英語で記述しているため)
- qwen2.5:7b のJSON生成が不安定な場合あり (リトライで対処)

### 次のステップ

- [ ] テストデータ生成完了を待つ (残り 4 Function + no_function)
- [ ] `python -m tools.evaluate` で全体評価を実行
- [ ] 評価結果に基づいてFunction description の改善を検討
- [ ] 必要に応じて FunctionGemma の Fine-tuning を検討
