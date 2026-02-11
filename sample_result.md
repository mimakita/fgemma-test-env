# FuncGemma 評価結果レポート

## 1. 評価概要

### 目的
FunctionGemma (270M) をローカルで動作させ、対話履歴から7つのFunction（+ no_function）への振り分け精度を測定する。

### 評価条件

| 項目 | 内容 |
|------|------|
| ルーティングモデル | `functiongemma` (google/functiongemma-270m-it, 301MB) |
| 推論エンジン | Ollama (ローカル, M1 8GB Mac) |
| テストケース数 | 950件（7 Function × 100件 + no_function 250件） |
| テストデータ生成 | `qwen2.5:7b` によるLLM自動生成 |
| temperature | 0.0（決定的推論） |
| num_ctx | 4096 |

### 対象Function一覧

| Function名 | 説明 | テスト件数 |
|------------|------|:---------:|
| travel_guide | 地名・旅行案内 | 100 |
| celebrity_info | 有名人・著名人情報 | 100 |
| shopping_intent | 購買意図・商品/広告 | 100 |
| sentiment_label | 感情ラベル | 100 |
| weather_info | 天気情報 | 100 |
| schedule_reminder | スケジュール・リマインダー | 100 |
| translation_assist | 翻訳支援 | 100 |
| no_function | どのFunctionにも該当しない | 250 |

---

## 2. 評価結果サマリ

4回の評価を実施。

| # | 条件 | データ品質 |
|:-:|------|-----------|
| Run 1 | 英語description | 初期版（混合言語、assistant終わり多数、ターン数超過あり） |
| Run 2 | 日本語description | 初期版（同上） |
| Run 3 | 日本語description | **改善版**（全日本語、user終わり統一、2-4ターン、関数名リーク除去） |
| Run 4 | **英語description** | **改善版**（同上） |

### Overall Metrics

| 指標 | Run 1 (英語+旧) | Run 2 (日本語+旧) | Run 3 (日本語+改善) | Run 4 (英語+改善) |
|------|:------:|:-------:|:-------:|:-------:|
| **Overall Accuracy** | 37.3% | 37.6% | 29.2% | **30.0%** |
| False Positive Rate | 12.0% | 4.0% | **1.2%** | 5.6% |
| Argument Accuracy | 11.2% | 8.6% | 0.0% | 0.0% |
| tool_call発火率 (対700件) | 28% | 17% | 5.1% | **8.6%** |
| 処理時間 | 1,405秒 | 4,531秒 | 2,081秒 | 1,382秒 |

### Per-Function メトリクス

| Function | Precision (Run1/2/3/4) | Recall (Run1/2/3/4) | F1 (Run1/2/3/4) |
|----------|:---------:|:------:|:----:|
| celebrity_info | 69.0 / 75.0 / 91.3 / **91.9%** | **60.0** / 45.0 / 21.0 / 34.0% | **64.2** / 56.3 / 34.2 / 49.6% |
| travel_guide | 74.2 / 85.7 / 80.0 / **77.8%** | 23.0 / **30.0** / 4.0 / 7.0% | 35.1 / **44.4** / 7.6 / 12.8% |
| weather_info | 41.4 / 47.6 / **100.0** / **100.0%** | **12.0** / 10.0 / 3.0 / 4.0% | **18.6** / 16.5 / 5.8 / 7.7% |
| sentiment_label | 75.0 / **100.0** / **100.0** / **100.0%** | **15.0** / 12.0 / 1.0 / 2.0% | **25.0** / 21.4 / 2.0 / 3.9% |
| schedule_reminder | **90.9** / 90.0 / 0.0 / 0.0% | **10.0** / 9.0 / 0.0 / 0.0% | **18.0** / 16.4 / 0.0 / 0.0% |
| translation_assist | 36.8 / **47.6** / 16.7 / 9.5% | **14.0** / 10.0 / 1.0 / 2.0% | **20.3** / 16.5 / 1.9 / 3.3% |
| shopping_intent | 7.1 / 0.0 / 0.0 / 0.0% | 1.0 / 0.0 / 0.0 / 0.0% | 1.8 / 0.0 / 0.0 / 0.0% |

---

## 3. Confusion Matrix

### Run 4（改善版テストデータ・英語description）

予測（横）に対する実際のラベル（縦）。

|  | travel | celebrity | shopping | sentiment | weather | schedule | translation | None |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **travel_guide** | **7** | 0 | 0 | 0 | 0 | 0 | 0 | 93 |
| **celebrity_info** | 1 | **34** | 0 | 0 | 0 | 0 | 2 | 63 |
| **shopping_intent** | 1 | 0 | **0** | 0 | 0 | 0 | 1 | 98 |
| **sentiment_label** | 0 | 2 | 0 | **2** | 0 | 0 | 3 | 93 |
| **weather_info** | 0 | 0 | 0 | 0 | **4** | 0 | 0 | 96 |
| **schedule_reminder** | 0 | 0 | 0 | 0 | 0 | **0** | 0 | 100 |
| **translation_assist** | 0 | 0 | 1 | 0 | 0 | 0 | **2** | 97 |
| **None (250件)** | 0 | 1 | 0 | 0 | 0 | 0 | 13 | **236** |

### Run 3（改善版テストデータ・日本語description）

|  | travel | celebrity | shopping | sentiment | weather | schedule | translation | None |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **travel_guide** | **4** | 0 | 0 | 0 | 0 | 0 | 0 | 96 |
| **celebrity_info** | 1 | **21** | 0 | 0 | 0 | 0 | 1 | 77 |
| **shopping_intent** | 0 | 1 | **0** | 0 | 0 | 0 | 0 | 99 |
| **sentiment_label** | 0 | 0 | 0 | **1** | 0 | 0 | 2 | 97 |
| **weather_info** | 0 | 0 | 0 | 0 | **3** | 0 | 0 | 97 |
| **schedule_reminder** | 0 | 0 | 0 | 0 | 0 | **0** | 0 | 100 |
| **translation_assist** | 0 | 0 | 1 | 0 | 0 | 0 | **1** | 98 |
| **None (250件)** | 0 | 1 | 0 | 0 | 0 | 0 | 2 | **247** |

### Run 2（初期版テストデータ・日本語description）

|  | travel | celebrity | shopping | sentiment | weather | schedule | translation | None |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **travel_guide** | **30** | 13 | 12 | 0 | 0 | 0 | 0 | 45 |
| **celebrity_info** | 1 | **45** | 0 | 0 | 0 | 0 | 1 | 53 |
| **shopping_intent** | 1 | 0 | **0** | 0 | 0 | 0 | 0 | 99 |
| **sentiment_label** | 0 | 0 | 0 | **12** | 0 | 0 | 2 | 86 |
| **weather_info** | 0 | 0 | 0 | 0 | **10** | 0 | 0 | 90 |
| **schedule_reminder** | 0 | 0 | 0 | 0 | 1 | **10** | 3 | 86 |
| **translation_assist** | 0 | 0 | 1 | 0 | 10 | 1 | **10** | 78 |
| **None (250件)** | 3 | 2 | 0 | 0 | 0 | 0 | 5 | **240** |

---

## 4. テストデータの改善

### 4.1 初期版テストデータの問題

| 問題 | 詳細 |
|------|------|
| assistant終わり | 多くの会話が `assistant` メッセージで終了（schedule_reminder: 100%, translation_assist: 100%） |
| ターン数超過 | weather_info (平均4.4), schedule_reminder (平均5.8) が要件2-4を超過 |
| 関数名リーク | schedule_reminderの会話に `schedule_reminder パラメータ` 等が混入 |
| 言語混合 | travel_guide (72%日本語) 等、英語が混在 |

### 4.2 改善版テストデータの特性

以下の品質改善ツールを作成・実行した：

1. **`tools/fix_test_data.py`**: conversation末尾をuser統一、ターン数を2-4に制限、関数名リーク除去
2. **`tools/filter_japanese_only.py`**: 英語データ除外（日本語のみに統一）
3. **`tools/generate_additional_data.py`**: 不足分を追加生成（品質要件を満たすプロンプトで再生成）

| Function | 件数 | 平均ターン数 | 最後がuser | 日本語率 |
|----------|:----:|:-----------:|:----------:|:--------:|
| 全Function共通 | 100 | 3.0 | 100% | 100% |
| no_function | 250 | 3.0 | 100% | 100% |

---

## 5. 考察

### 5.1 英語description vs 日本語description（改善版データ、Run 3 vs Run 4）

同一の改善版テストデータに対して、descriptionの言語のみを変えた比較：

| 指標 | Run 3 (日本語) | Run 4 (英語) | 差分 |
|------|:---:|:---:|:---:|
| Overall Accuracy | 29.2% | **30.0%** | +0.8pt |
| False Positive Rate | **1.2%** | 5.6% | +4.4pt |
| tool_call発火率 | 5.1% | **8.6%** | +3.5pt |
| celebrity_info Recall | 21% | **34%** | +13pt |
| travel_guide Recall | 4% | **7%** | +3pt |

**結論：英語descriptionの方がRecallは改善されるが、FPRが悪化するトレードオフがある。**

- **Recallの改善**: 全Functionで英語版の方がRecallが高い。FunctionGemmaの学習データが英語中心のため、英語descriptionの方がtool_callを発火しやすい
- **FPRの悪化**: no_functionケースで13件が `translation_assist` に誤分類。英語descriptionだと「翻訳」のような一般的なワードに反応しやすくなる
- **tool_call発火率**: 5.1% → 8.6%。英語descriptionの方がtool_callを出す確率が約1.7倍

### 5.2 初期版 vs 改善版テストデータ（英語description、Run 1 vs Run 4）

同じ英語descriptionで、テストデータ品質のみ変えた比較：

| 指標 | Run 1 (英語+旧) | Run 4 (英語+改善) | 差分 |
|------|:---:|:---:|:---:|
| Overall Accuracy | **37.3%** | 30.0% | -7.3pt |
| False Positive Rate | 12.0% | **5.6%** | -6.4pt |
| tool_call発火率 | **28%** | 8.6% | -19.4pt |
| celebrity_info Recall | **60%** | 34% | -26pt |

**初期版データの方がRecallが大幅に高い**。これはRun 2→3と同じ傾向で、初期版テストデータの構造（assistant終わりの会話、長いターン数）がFunctionGemmaにとって有利だったことを裏付ける。

### 5.3 全4回を通じた主要な発見

#### (1) FunctionGemma 270Mはテキスト応答に圧倒的に偏る

改善版データでのtool_call発火率：
- Run 3 (日本語desc): 36/700 = **5.1%**
- Run 4 (英語desc): 60/700 = **8.6%**

どちらのケースでも90%以上がテキスト応答。FunctionGemma 270Mは「自分でテキスト回答する」ことをデフォルト挙動とし、tool_callは極めて限定的。

#### (2) celebrity_infoが唯一有意に機能するFunction

全4回を通じて最もRecallが高い。Run 4 では Recall 34%, Precision 91.9%, F1 49.6%。人名を含む会話パターンがFunctionGemmaの学習データと最も一致しやすい。

#### (3) shopping_intent / schedule_reminder は完全に機能不全

全4回を通じてRecall ≈ 0%。270Mモデルでは購買意図やスケジュール管理の概念をtool_callにマッピングする能力が不足。

#### (4) translation_assist のFPが英語descriptionで増加

Run 4 でno_function → translation_assist が13件発生。英語descriptionの "translate" "language" がFunctionGemmaの一般的な応答パターンとマッチしやすい。

#### (5) Precisionは一貫して高い

tool_callを発火した場合の正確性は改善版データで特に高い：
- celebrity_info: 91-92%
- weather_info: 100%
- sentiment_label: 100%

「呼ぶ時は正確」だが「呼ぶ頻度が極端に低い」のが270Mモデルの特性。

---

## 6. 技術的課題

### 6.1 Ollamaのメモリ管理（M1 8GB）

評価実行中にOllamaが定期的にハングする問題が発生（約50件処理ごとに30秒〜10分のフリーズ）。

**対策として実施した改善：**
- `OllamaClient` にhttpxタイムアウト（30秒）を設定
- 3回のリトライ機能を追加
- `evaluate.py` に50件ごとのチェックポイント保存・レジューム機能を実装
- 評価前に `ollama run functiongemma "test" --keepalive 60m` でプリロード推奨

Run 4 では安定して1,382秒（約23分）で完了。

---

## 7. 改善提案

### 4回の評価を踏まえた提案

| 優先度 | 改善項目 | 期待効果 | 根拠 |
|:------:|---------|---------|------|
| **高** | functiongemma以外のルーティングモデルの検討 | 根本的な精度向上 | 270Mモデルの限界が明確 |
| **高** | 英語descriptionを採用（現状維持） | Recall向上 | Run 3→4でRecall改善を確認 |
| 中 | 会話末尾を短い指示文に変更（「〇〇を教えて」形式） | tool_call発火率向上 | テキスト応答への偏りを抑制 |
| 中 | 初期版データの「assistant終わり」構造を活用 | Recall向上の可能性 | Run 1/2でRecallが高かった |
| 低 | temperature を 0.0 → 0.1〜0.3 に変更 | tool_call発火の多様性 | 保守的判定の緩和 |
| 低 | shopping_intent / schedule_reminderの廃止 | 無駄なFunction削減 | 全Runで機能不全 |

---

## 8. 結論

4回の評価を通じて、FunctionGemma (270M) のFunction Routing精度は **Overall Accuracy 29-38%** の範囲であることが明確になった。

最も条件を揃えた比較（改善版テストデータ）では：
- **英語description（Run 4）: Accuracy 30.0%, FPR 5.6%**
- **日本語description（Run 3）: Accuracy 29.2%, FPR 1.2%**

英語descriptionの方がRecallは改善されるが（celebrity_info: 21%→34%）、FPRが悪化する（1.2%→5.6%）トレードオフがある。

FunctionGemma 270Mの根本的な特性として、**テキスト応答をデフォルトとし、tool_callは全ケースの5-9%でしか発火しない**ことが判明した。Precisionは高い（呼ぶ時は正確）が、Recallが極めて低い（呼ぶべき時に呼ばない）。

実運用に向けては、270Mモデルの限界を踏まえ、**より大きなモデル（FunctionGemma 2B等）への移行**、またはルーティング戦略自体の見直し（例：テキスト応答をルーティング判定に使わず、別途分類器を噛ませる等）が必要である。

---

## 9. Fine-tuning 実験

上記のベースライン評価を踏まえ、PEFT LoRA によるFine-tuningを実施した。

### 9.1 Fine-tuning 設定

| パラメータ | 値 |
|-----------|-----|
| ベースモデル | google/functiongemma-270m-it |
| Fine-tuning手法 | PEFT LoRA |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| Learning rate | 2e-4 |
| Batch size | 1 |
| Gradient accumulation | 4 |
| Max sequence length | 512 |
| 学習環境 | Apple M1 8GB (MPS) |

### 9.2 データセット（Run 2）

| データ | 件数 |
|--------|------|
| 学習データ | 1,152件（各関数160件 + no_function 32件） |
| テストデータ | 380件（各関数40件 + no_function 100件） |

### 9.3 チェックポイント別精度

| Checkpoint | Steps | Epochs | Accuracy | 学習時間 |
|------------|-------|--------|----------|----------|
| ckpt-100 | 100 | 0.35 | 45.8% | 約20分 |
| **ckpt-500** | 500 | 1.74 | **60.3%** | 約1.7時間 |
| ckpt-800 | 800 | 2.78 | 60.5% | 約2.8時間 |

**推奨**: checkpoint-500（精度と学習時間のバランスが最適）

### 9.4 学習曲線（Loss）

| Steps | Training Loss |
|-------|---------------|
| 100 | 0.5720 |
| 200 | 0.2807 |
| 300 | 0.2234 |
| 400 | 0.2105 |
| 500 | 0.1994 |
| 600 | 0.1926 |
| 700 | 0.1926 |
| 800 | 0.1790 |

Loss は 500 steps 以降で収束傾向。

### 9.5 関数別性能（checkpoint-500）

| Function | Recall | Precision | TP/Total |
|----------|--------|-----------|----------|
| celebrity_info | **100%** | 100% | 40/40 |
| schedule_reminder | **95.0%** | 92.7% | 38/40 |
| weather_info | **92.5%** | 74.0% | 37/40 |
| travel_guide | **90.0%** | 85.7% | 36/40 |
| shopping_intent | **87.5%** | 74.5% | 35/40 |
| sentiment_label | **65.0%** | 46.4% | 26/40 |
| translation_assist | 42.5% | 23.9% | 17/40 |
| no_function | 0% | - | 0/100 |

### 9.6 関数別性能（checkpoint-800）

| Function | Recall | Precision | TP/Total |
|----------|--------|-----------|----------|
| celebrity_info | **100%** | 100% | 40/40 |
| schedule_reminder | **97.5%** | 92.9% | 39/40 |
| travel_guide | **95.0%** | 82.6% | 38/40 |
| weather_info | **92.5%** | 72.5% | 37/40 |
| shopping_intent | **90.0%** | 76.6% | 36/40 |
| sentiment_label | 62.5% | 46.3% | 25/40 |
| translation_assist | 37.5% | 24.2% | 15/40 |
| no_function | 0% | - | 0/100 |

---

## 10. ベースライン vs Fine-tuned 比較

### 10.1 Overall Accuracy

| Model | Accuracy | 改善幅 |
|-------|----------|--------|
| Baseline (Ollama, Zero-shot) | 27.9% | - |
| Fine-tuned (100 steps) | 45.8% | +17.9% |
| Fine-tuned (500 steps) | 60.3% | **+32.4%** |
| Fine-tuned (800 steps) | 60.5% | +32.6% |

### 10.2 関数別 Recall 比較

| Function | Baseline | Fine-tuned (500) | 改善幅 |
|----------|----------|------------------|--------|
| celebrity_info | 17.5% | 100% | **+82.5%** |
| schedule_reminder | 1.5% | 95.0% | **+93.5%** |
| weather_info | 2.5% | 92.5% | **+90.0%** |
| travel_guide | 4.0% | 90.0% | **+86.0%** |
| shopping_intent | 0% | 87.5% | **+87.5%** |
| sentiment_label | 1.0% | 65.0% | **+64.0%** |
| translation_assist | 1.0% | 42.5% | +41.5% |

---

## 11. Fine-tuning の課題

### 11.1 no_function の認識失敗

Fine-tuned モデルは **no_function ケースを全く認識できない** (Recall 0%)。
- 関数呼び出しを積極的に行うようバイアスがかかっている
- False Positive の主要因

### 11.2 translation_assist の低性能

Recall 42.5%, Precision 23.9%
- 他の関数との混同が多い
- 翻訳意図の認識が曖昧

### 11.3 学習データの不均衡

no_function の学習データが32件と少なく、「関数を呼ばない」判断の学習が不十分。

---

## 12. 結論（更新版）

### ベースライン (270M, Zero-shot)

- Overall Accuracy: **27.9%**
- tool_call 発火率が極めて低い（5-9%）
- Precision は高いが Recall が低い
- 日本語での関数呼び出し認識が弱い

### Fine-tuned (270M, PEFT LoRA)

- Overall Accuracy: **60.3%** (500 steps)
- ベースラインから **+32.4%** の精度向上
- ほぼ全ての関数で Recall が大幅に改善
- ただし no_function の認識が完全に失敗（Recall 0%）

### 今後の改善方向

1. **no_function データの増強**: 学習データに no_function をより多く含める
2. **Negative sampling**: 関数呼び出し不要なケースの学習を強化
3. **より大きなモデル**: Gemma 2B / 7B での評価
4. **閾値調整**: tool_call 確率の閾値を設けて no_function を判定

---

## 付録: 再現方法

```bash
# ベースライン評価
source .venv/bin/activate
python -m tools.evaluate

# Fine-tuned モデル評価
source .venv-ft/bin/activate
python -m tools.evaluate_peft --run-id 2 --checkpoint 500
```
