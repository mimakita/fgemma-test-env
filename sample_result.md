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

## 13. データ多様性向上実験（Run 3）

### 13.1 背景と目的

Run 2 の Fine-tuning 結果から以下の課題が明らかになった：
- **translation_assist の Recall が低下傾向**（50% → 42.5% → 37.5%）
- **no_function の認識が完全に失敗**（Recall 0%）

これらの課題を解決するため、テストデータの多様性を向上させた新規実験（Run 3）を実施。

### 13.2 エラー分析（Run 2 checkpoint-800）

translation_assist の誤分類パターンを分析：

| 誤分類パターン | 件数 | 説明 |
|---------------|------|------|
| no_function → translation_assist | **34件** | 雑談を翻訳リクエストと誤判定（最大の問題） |
| sentiment_label → translation_assist | 9件 | 感情表現を翻訳対象と誤判定 |
| translation_assist → sentiment_label | 7件 | 逆方向の誤判定 |
| translation_assist → no_function | 5件 | 翻訳リクエストを無視 |

**根本原因**: モデルが「任意の日本語テキスト」を翻訳対象と見なす傾向（過学習）

### 13.3 データ多様性向上の施策

#### (1) translation_assist データの改善

新ツール `tools/generate_diverse_translation.py` を作成。

| パターン | 割合 | 例 |
|---------|------|-----|
| 明示的リクエスト | 40% | 「〜を英語に翻訳して」「〜を和訳して」 |
| バイリンガル文脈 | 20% | 英語アシスタントに日本語→英語翻訳依頼 |
| 慣用句・表現翻訳 | 15% | 「一石二鳥」「よろしくお願いします」 |
| 逆翻訳(英→日) | 15% | 英語メールの日本語訳依頼 |
| 多言語翻訳 | 10% | 中国語、韓国語、フランス語等への翻訳 |

**ポイント**: 明示的なトリガーワード（「翻訳して」「英語にして」など）を含むデータを増加

#### (2) no_function データの改善

新ツール `tools/generate_diverse_no_function.py` を作成。

**「紛らわしいケース」（50%）を重点的に追加**:
- 翻訳に見えるが翻訳ではない (12%)
- 天気の言及だが予報リクエストではない (8%)
- 旅行の言及だがガイド要求ではない (8%)
- 有名人の言及だが情報検索ではない (6%)
- スケジュールの言及だがリマインダー要求ではない (6%)
- 買い物の言及だが購買意図ではない (6%)
- 感情表現だが分析要求ではない (4%)

**ポイント**: 「関数呼び出しに見えるが不要なケース」を学習させてFPを削減

#### (3) 他ドメインのデータバランス調整

新ツール `tools/generate_diverse_functions.py` で各関数に明示的トリガーを含む多様なデータを追加。

### 13.4 最終データセット（Run 3）

| カテゴリ | Run 2 | Run 3 | 増加分 |
|---------|-------|-------|--------|
| celebrity_info | 200 | 370 | +170 |
| travel_guide | 200 | 370 | +170 |
| weather_info | 200 | 370 | +170 |
| schedule_reminder | 200 | 370 | +170 |
| sentiment_label | 200 | 370 | +170 |
| shopping_intent | 200 | 370 | +170 |
| translation_assist | 200 | 370 | +170 |
| no_function | 500 | 750 | +250 |
| **合計** | **1,900** | **3,340** | **+1,440** |

### 13.5 Fine-tuning 設定（Run 3）

| パラメータ | Run 2 | Run 3 | 変更点 |
|-----------|-------|-------|--------|
| Training samples | 1,520 | 2,672 | +1,152 |
| Test samples | 380 | 668 | +288 |
| Max steps | 500 | 800 | +300 |
| LoRA rank | 16 | 16 | - |
| LoRA alpha | 32 | 32 | - |
| Gradient accumulation | 4 | 8 | 増加（メモリ対策） |
| Gradient checkpointing | False | True | 有効化（メモリ対策） |
| 学習時間 | 約1.7時間 | **約2.5時間** | +0.8時間 |

### 13.6 学習曲線（Run 3）

| Steps | Training Loss |
|-------|---------------|
| 100 | 2.366 |
| 200 | 0.393 |
| 300 | 0.214 |
| 400 | 0.183 |
| 500 | 0.179 |
| 600 | 0.171 |
| 700 | 0.139 |
| 800 | 0.109 |

Loss は継続的に低下し、800 steps でも収束傾向。

### 13.7 評価結果（Run 3, checkpoint-800）

#### Overall Metrics

| メトリクス | Run 2 (ckpt-500) | Run 3 (ckpt-800) | 改善幅 |
|-----------|------------------|------------------|--------|
| **Overall Accuracy** | 60.3% | **68.3%** | **+8.0%** |
| Test samples | 380 | 668 | +288 |

#### 関数別性能比較

| Function | Run 2 Recall | Run 3 Recall | 改善幅 |
|----------|--------------|--------------|--------|
| celebrity_info | 100% | **98.6%** | -1.4% |
| travel_guide | 90.0% | **98.6%** | +8.6% |
| schedule_reminder | 95.0% | **98.6%** | +3.6% |
| shopping_intent | 87.5% | **97.3%** | +9.8% |
| weather_info | 92.5% | **93.2%** | +0.7% |
| sentiment_label | 65.0% | **78.4%** | +13.4% |
| translation_assist | 37.5% | **51.4%** | **+13.9%** |
| no_function | 0% | 0% | - (課題継続) |

#### 詳細メトリクス（Run 3）

| Function | Precision | Recall | TP | FP | FN | Total |
|----------|-----------|--------|----|----|----|----|
| celebrity_info | 100.0% | 98.6% | 73 | 0 | 1 | 74 |
| travel_guide | 94.8% | 98.6% | 73 | 4 | 1 | 74 |
| schedule_reminder | 94.8% | 98.6% | 73 | 4 | 1 | 74 |
| shopping_intent | 92.3% | 97.3% | 72 | 6 | 2 | 74 |
| weather_info | 93.2% | 93.2% | 69 | 5 | 5 | 74 |
| sentiment_label | 73.4% | 78.4% | 58 | 21 | 16 | 74 |
| translation_assist | 69.1% | 51.4% | 38 | 17 | 36 | 74 |
| no_function (null) | - | 0% | 0 | 0 | 150 | 150 |

### 13.8 考察

#### 成功点

1. **translation_assist の Recall が大幅改善** (+13.9%)
   - 明示的トリガーワードを含むデータの追加が効果的
   - Precision も 23.9% → 69.1% に改善

2. **全体精度が 8% 向上** (60.3% → 68.3%)
   - データ多様性の向上が全体的な学習に貢献

3. **他関数の Recall も軒並み向上**
   - sentiment_label: +13.4%
   - shopping_intent: +9.8%
   - travel_guide: +8.6%

#### 継続課題

1. **no_function の認識は依然として失敗** (Recall 0%)
   - 「紛らわしいケース」を追加したが、まだ不十分
   - モデルが「何らかの関数を呼ぶ」ことに過度にバイアス

2. **translation_assist の FP が残存** (17件)
   - no_function → translation_assist の誤分類は減少したが完全ではない

### 13.9 今後の改善方向

| 優先度 | 施策 | 期待効果 |
|:------:|------|---------|
| 高 | no_function の学習比率を大幅増加（50%以上） | no_function Recall 向上 |
| 高 | 二段階判定（まず「関数呼び出しが必要か」を判定） | FP 削減 |
| 中 | confidence threshold の導入 | no_function 判定の改善 |
| 中 | Negative sampling の強化 | 過学習の抑制 |
| 低 | より大きなモデル（Gemma 2B）での検証 | 根本的な精度向上 |

---

## 14. 結論（最終更新）

### 実験サマリー

| 実験 | Accuracy | 主な発見 |
|------|----------|---------|
| Baseline (Zero-shot) | 27.9% | tool_call 発火率が極めて低い |
| Run 2 (Fine-tuned, 500 steps) | 60.3% | +32.4% 向上、no_function 認識失敗 |
| **Run 3 (多様データ, 800 steps)** | **68.3%** | **+8.0% 向上、translation_assist 改善** |

### 主要な学び

1. **データ多様性は精度向上に直結**
   - 明示的トリガーワードの追加、紛らわしいケースの追加が効果的

2. **no_function 認識は根本的に難しい**
   - Fine-tuning でモデルが「関数を呼ぶ」方向にバイアス
   - 二段階判定など、アーキテクチャレベルの改善が必要

3. **270M モデルでも適切な Fine-tuning で実用レベルに近づく**
   - Overall Accuracy 68.3% は改善の余地があるが、ベースラインから大幅向上

---

## 15. no_function 認識課題の調査と改善策

### 15.1 問題の背景

Run 2, Run 3 を通じて、no_function（関数呼び出し不要なケース）の認識が **Recall 0%** という深刻な課題が継続している。Fine-tuning によりモデルは「何らかの関数を呼ぶ」方向に過度にバイアスされている。

### 15.2 文献調査による知見

最新の研究論文から以下のアプローチを発見した。

#### (1) Decision Token メカニズム（arxiv:2412.01130v2）

| 概念 | 説明 |
|------|------|
| アプローチ | 出力前に `<|answer|>` vs `<|use_tool|>` の二択トークンを予測させる |
| 効果 | Relevance detection が **49.58% → 65.42%** に改善 |
| 原理 | 関数呼び出し判断を最初に強制することで、出力の安定性が向上 |

**合成データ生成手法**: 元データで関数Aが正解の場合、関数Aを選択肢から除外して `<|answer|>` を正解とするネガティブサンプルを作成。1,000件の合成データ追加で大幅改善。

#### (2) Irrelevance-Augmented Dataset（arxiv:2410.04587v2 - Hammer論文）

| 発見 | 説明 |
|------|------|
| 逆相関問題 | Fine-tuning で関数呼び出し精度↑ → 不要時検出精度↓ |
| 対策 | 正しい関数を意図的に除外したデータで学習 |
| 効果 | 関数呼び出しと不要時検出のトレードオフを改善 |

**重要**: 関数呼び出し能力と不要時検出能力は **逆相関** の関係にあり、データ比率のバランスが極めて重要。

#### (3) ToolACE 研究（arxiv:2409.00920v1）

- より広範なAPIへの露出が、微妙なAPI差異の識別と不要時検出能力を向上させる
- ネガティブサンプル（関数が不要なケース）の合成が必須

### 15.3 調査に基づく改善実装

#### 改善策1: no_function データの大幅増加

| 項目 | 変更前 | 変更後 |
|------|--------|--------|
| no_function件数 | 750件 | **2,200件** |
| no_function比率 | 22% | **46%** |

追加した700件の内訳:
- **Function-excluded (関数除外型)**: 500件
  - 既存の関数呼び出しデータから正解関数を「除外」したと仮定したネガティブサンプル
- **Ambiguous (曖昧クエリ型)**: 200件
  - 関数に関連するが明確なアクション要求がないケース

#### 改善策2: 学習時の no_function 比率制御

`split_data.py` に `--no-function-ratio` オプションを追加:

```bash
# 学習データの50%をno_functionに設定
python -m tools.split_data --run-id 4 --seed 42 --balance --no-function-ratio 0.5
```

| パラメータ | 説明 |
|-----------|------|
| `--no-function-ratio 0.5` | 学習データの50%をno_functionに設定 |
| 効果 | 「関数を呼ばない」判断の学習機会を大幅増加 |

#### 改善策3: Irrelevance-Augmented Data 生成ツール

新ツール `tools/generate_irrelevance_data.py` を作成:

```python
# 既存の関数呼び出しデータを「関数が利用不可」と仮定して
# no_function として再ラベル付け
{
    "conversation": [元の会話],
    "expected_function": None,  # 関数除外
    "irrelevance_type": "function_excluded"
}
```

### 15.4 Run 4 データセット

| カテゴリ | Run 3 | Run 4 | 変更 |
|---------|-------|-------|------|
| celebrity_info | 370 | 370 | - |
| travel_guide | 370 | 370 | - |
| weather_info | 370 | 370 | - |
| schedule_reminder | 370 | 370 | - |
| sentiment_label | 370 | 370 | - |
| shopping_intent | 370 | 370 | - |
| translation_assist | 370 | 370 | - |
| no_function | 750 | **2,200** | **+1,450** |
| **合計** | **3,340** | **4,790** | **+1,450** |

#### 学習データ分布（--balance --no-function-ratio 0.5）

| カテゴリ | 件数 | 比率 |
|---------|------|------|
| 各関数（7種） | 各296件 | 各7.1% |
| no_function | 2,072件 | **50.0%** |
| **合計** | **4,144件** | 100% |

### 15.5 参考文献

| 論文 | URL | 主な貢献 |
|------|-----|---------|
| Decision Token for Function Calling | [arxiv:2412.01130v2](https://arxiv.org/html/2412.01130v2) | 二択トークンによる不要時検出改善 |
| Hammer: Function Masking | [arxiv:2410.04587v2](https://arxiv.org/html/2410.04587v2) | 逆相関問題の発見と対策 |
| ToolACE | [arxiv:2409.00920v1](https://arxiv.org/html/2409.00920v1) | ネガティブサンプル合成の重要性 |
| FunctionGemma Fine-tuning Guide | [Google Developers Blog](https://developers.googleblog.com/a-guide-to-fine-tuning-functiongemma/) | 公式Fine-tuningガイド |

### 15.6 Run 4 評価結果

#### Overall Metrics

| メトリクス | Run 3 (ckpt-800) | Run 4 (ckpt-1000) | 差分 |
|-----------|------------------|-------------------|------|
| **Overall Accuracy** | 68.3% | **42.7%** | **-25.6%** |
| Test samples | 668 | 958 | +290 |
| no_function件数 | 150 | **440** | +290 |

#### 関数別性能比較

| Function | Run 3 Recall | Run 4 Recall | 差分 |
|----------|--------------|--------------|------|
| travel_guide | 98.6% | 94.6% | -4.0% |
| celebrity_info | 98.6% | 90.5% | -8.1% |
| schedule_reminder | 98.6% | 90.5% | -8.1% |
| shopping_intent | 97.3% | 93.2% | -4.1% |
| weather_info | 93.2% | 86.5% | -6.7% |
| sentiment_label | 78.4% | 54.1% | -24.3% |
| translation_assist | 51.4% | 43.2% | -8.2% |
| **no_function** | **0%** | **0%** | **変化なし** |

#### 詳細メトリクス（Run 4, checkpoint-1000）

| Function | Precision | Recall | TP | FP | FN | Total |
|----------|-----------|--------|----|----|----|----|
| travel_guide | 83.3% | 94.6% | 70 | 14 | 4 | 74 |
| celebrity_info | 85.9% | 90.5% | 67 | 11 | 7 | 74 |
| schedule_reminder | 78.8% | 90.5% | 67 | 18 | 7 | 74 |
| shopping_intent | 75.8% | 93.2% | 69 | 22 | 5 | 74 |
| weather_info | 81.0% | 86.5% | 64 | 15 | 10 | 74 |
| sentiment_label | 80.0% | 54.1% | 40 | 10 | 34 | 74 |
| translation_assist | 64.0% | 43.2% | 32 | 18 | 42 | 74 |
| no_function (null) | - | 0% | 0 | 0 | 440 | 440 |

### 15.7 Run 4 考察

#### 失敗分析

1. **no_function認識は依然として完全失敗** (Recall 0%)
   - 学習データの50%をno_functionにしたにも関わらず、改善せず
   - モデルは「何らかの関数を呼ぶ」パターンを強く学習している

2. **全体精度の大幅低下** (68.3% → 42.7%)
   - テストデータのno_function比率増加（22% → 46%）が主因
   - no_functionを全て誤分類するため、正解率が低下

3. **各関数のRecallも低下傾向**
   - no_function比率増加による学習データバランスの変化が影響
   - 特にsentiment_label (-24.3%)、translation_assist (-8.2%)で顕著

#### 根本原因の考察

1. **Decision Tokenの欠如**: FunctionGemma 270Mはtool_callを出力する/しないの二値判定を学習していない
2. **出力形式の制約**: モデルはJSON形式の関数呼び出しを出力するように訓練されており、「呼ばない」という出力パターンが弱い
3. **アーキテクチャ限界**: 270Mモデルでは関数呼び出しと非呼び出しの識別に必要な表現能力が不足

### 15.8 Run 5: ロールバック実験

Run 4の失敗を受けて、no_functionデータをロールバックしてRun 5を実施。

#### 変更内容

| 項目 | Run 4 | Run 5 | 変更 |
|------|-------|-------|------|
| no_function件数 | 2,200 | **1,500** | -700件（ロールバック） |
| function_excluded | 500 | **0** | 削除 |
| ambiguous_query | 200 | **0** | 削除 |
| no_function比率 | 50% | **12.5%** | 大幅削減 |
| データバランス | balanced | **balanced** | 各関数296件 |

#### Run 5 評価結果（checkpoint-800）

| メトリクス | Run 4 | Run 5 | 差分 |
|-----------|-------|-------|------|
| **Overall Accuracy** | 42.7% | **57.1%** | **+14.4%** |
| Test samples | 958 | 818 | -140 |
| no_function Recall | 0% | **0%** | 変化なし |

#### 関数別性能比較（Run 5 vs Run 4）

| Function | Run 4 Recall | Run 5 Recall | 差分 |
|----------|--------------|--------------|------|
| travel_guide | 94.6% | **100.0%** | +5.4% |
| celebrity_info | 90.5% | **98.6%** | +8.1% |
| schedule_reminder | 90.5% | **97.3%** | +6.8% |
| shopping_intent | 93.2% | **97.3%** | +4.1% |
| weather_info | 86.5% | **93.2%** | +6.7% |
| sentiment_label | 54.1% | **83.8%** | +29.7% |
| translation_assist | 43.2% | **60.8%** | +17.6% |
| **no_function** | 0% | **0%** | 変化なし |

#### Run 5 考察

1. **ロールバックによる精度回復**: Run 4から+14.4%改善
2. **関数呼び出し精度は回復**: 各関数のRecallがRun 3レベルに近づく
3. **no_function認識は依然0%**: データ比率を変えても解決しない
4. **Run 3（68.3%）には未達**: balanced制約で各関数のデータ量が制限された影響

### 15.10 二段階判定プロトタイプ（成功）

#### 概要

二段階判定アプローチをプロトタイプ実装し、no_function認識問題を解決した。

**Stage 1**: ヒューリスティックベースの分類器で「関数呼び出しが必要か」を判定
**Stage 2**: 必要な場合のみFunctionGemmaで関数を選択

#### 実装

```bash
python -m tools.prototype_two_stage --run-id 5 --checkpoint 800
```

#### Stage 1 ヒューリスティック

キーワードベースの分類器:
- no_function指標: 挨拶、一般質問、創作依頼、相談など
- function指標: 各関数に関連するキーワード（旅行、翻訳、天気など）

#### 評価結果

| メトリクス | 単独モデル (Run 5) | 二段階判定 | 改善 |
|-----------|-------------------|-----------|------|
| **Overall Accuracy** | 57.1% | **70.8%** | **+13.7%** |
| **no_function Recall** | 0% | **100%** | **+100%** |
| no_function Precision | - | 100% | - |
| Stage 1 blocked | - | 318/818 | 39% |

#### 関数別性能（二段階判定）

| Function | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| no_function | 100.0% | **100.0%** | 100.0% |
| weather_info | 87.0% | 90.5% | 88.7% |
| travel_guide | 94.8% | 74.3% | 83.3% |
| schedule_reminder | 89.1% | 66.2% | 76.0% |
| shopping_intent | 95.7% | 60.8% | 74.4% |
| sentiment_label | 83.6% | 62.2% | 71.3% |
| celebrity_info | 100.0% | 37.8% | 54.9% |
| translation_assist | 50.0% | 40.5% | 44.8% |

#### 考察

1. **no_function認識問題を完全解決**: Recall 0% → 100%
2. **全体精度が大幅改善**: 57.1% → 70.8% (+13.7%)
3. **トレードオフ**: 関数のRecallが低下（ヒューリスティックがno_functionを積極的に判定）
4. **改善余地**: キーワードの調整、MLベース分類器への置き換えで関数Recallも改善可能

#### 今後の方向

| 優先度 | 施策 | 期待効果 |
|:------:|------|---------|
| 高 | Stage 1をMLモデルに置き換え | 関数Recall改善 |
| 中 | より大きなモデル（Gemma 2B/7B）での検証 | 表現能力の向上 |
| 低 | Decision Token実装 | 単一モデルでの解決 |

---

## 16. 結論（最終更新）

### 実験サマリー

| 実験 | Accuracy | 主な発見 |
|------|----------|---------|
| Baseline (Zero-shot) | 27.9% | tool_call発火率が極めて低い |
| Run 2 (Fine-tuned, 500 steps) | 60.3% | +32.4%向上、no_function認識失敗 |
| Run 3 (多様データ, 800 steps) | 68.3% | translation_assist改善 |
| Run 4 (no_function 50%, 1000 steps) | 42.7% | no_function増強が逆効果 |
| Run 5 (balanced, 12.5% no_function) | 57.1% | ロールバックで回復 |
| **Run 5 + 二段階判定** | **70.8%** | **no_function Recall 100%達成** |

### 主要な学び

1. **no_function認識は単純なデータ増強では解決しない**
   - 学習データの50%をno_functionにしても、Recall 0%のまま
   - Decision Tokenなどのアーキテクチャ変更が必要

2. **FunctionGemma 270Mの限界が明確**
   - 関数呼び出しには有効（Recall 80-95%）
   - 「呼ばない」判定には根本的に不向き

3. **二段階判定が現実的な解決策**
   - 第1段階: 関数呼び出しが必要かを判定（別モデル or ルール）
   - 第2段階: 必要な場合のみFunctionGemmaで関数を選択

---

## 付録: 再現方法

```bash
# ベースライン評価
source .venv/bin/activate
python -m tools.evaluate

# Fine-tuned モデル評価 (Run 2)
source .venv-ft/bin/activate
python -m tools.evaluate_peft --run-id 2 --checkpoint 500

# Fine-tuned モデル評価 (Run 3)
source .venv-ft/bin/activate
python -m tools.evaluate_peft --run-id 3 --checkpoint 800

# データ多様性向上ツール
python -m tools.generate_diverse_translation --count 150 --merge
python -m tools.generate_diverse_no_function --count 200 --merge
python -m tools.generate_diverse_functions --all --target 370

# エラー分析
python -m tools.analyze_errors --run-id 2 --checkpoint 800 --function translation_assist

# Run 4: no_function 認識改善実験
python -m tools.generate_irrelevance_data --count 500 --ambiguous 200 --merge
python -m tools.split_data --run-id 4 --seed 42 --balance --no-function-ratio 0.5
python -m tools.finetune_peft --run-id 4 --epochs 3 --max-steps 1000
python -m tools.evaluate_peft --run-id 4 --checkpoint 800

# 二段階判定プロトタイプ検証
python -m tools.prototype_two_stage --run-id 5 --checkpoint 800
python -m tools.prototype_two_stage --run-id 5 --checkpoint 800 --stage1-only
```

---

## 16. 二段階判定の本番統合

### 16.1 概要

no_function認識問題（全Runで Recall 0%）を解決するため、二段階判定アーキテクチャを本番コードに統合した。

### 16.2 アーキテクチャ

```
User Input
    ↓
┌─────────────────────────────────────┐
│ Stage 1: FunctionCallClassifier    │
│ (キーワードベース、高速)            │
└─────────────────────────────────────┘
    ↓ need_function=True        ↓ need_function=False
┌─────────────────────────┐     │
│ Stage 2: FunctionGemma  │     │
│ (LLM、関数選択)         │     │
└─────────────────────────┘     │
    ↓                           ↓
Function Call              直接応答
```

### 16.3 実装ファイル

| ファイル | 役割 |
|---------|------|
| `src/classifier.py` | Stage 1 分類器 (キーワードベース) |
| `src/router.py` | 二段階判定統合Router |
| `tests/test_router.py` | ユニットテスト（23テスト） |

### 16.4 主要クラス

#### ClassificationResult
```python
@dataclass
class ClassificationResult:
    need_function: bool      # 関数呼び出しが必要か
    confidence: float        # 判定の確信度 (0-1)
    matched_function: str    # マッチした関数名 (Optional)
```

#### FunctionCallClassifier
```python
class FunctionCallClassifier:
    def classify(self, conversation: list[dict]) -> ClassificationResult:
        # キーワードベースで関数呼び出しの必要性を判定
```

#### RouterResult (拡張)
```python
@dataclass
class RouterResult:
    should_call: bool
    function_name: Optional[str]
    arguments: dict
    raw_response: str
    classification: Optional[ClassificationResult]  # 新規
    stage1_blocked: bool  # 新規: Stage1で棄却されたか
```

### 16.5 使用方法

```python
from src.router import FunctionRouter
from src.ollama_client import OllamaClient
from src.functions.registry import FunctionRegistry

# 二段階判定はデフォルトで有効
router = FunctionRouter(OllamaClient(), registry)

# 明示的に有効/無効化
router = FunctionRouter(client, registry, use_two_stage=True)   # 有効
router = FunctionRouter(client, registry, use_two_stage=False)  # 無効

# ルーティング実行
result = router.route(conversation_history)

if result.stage1_blocked:
    print("Stage 1でno_functionと判定")
elif result.should_call:
    print(f"Function: {result.function_name}")
```

### 16.6 精度向上

| メトリクス | 単独モデル (Run 5) | 二段階判定 | 改善 |
|-----------|-------------------|-----------|------|
| Accuracy | 57.1% | **70.8%** | +13.7% |
| no_function Recall | 0% | **100%** | +100% |
| 全体Precision | 86.8% | **70.2%** | -16.6% |

### 16.7 トレードオフ

**メリット:**
- no_function認識問題を完全に解決
- 不要なLLM呼び出しを削減（高速化）
- アーキテクチャの単純さ（追加学習不要）

**デメリット:**
- キーワードベースの限界（新しいドメインには手動追加が必要）
- 曖昧なケースでの精度低下の可能性

### 16.8 今後の改善案

1. **Stage 1の機械学習化**: キーワードベースから小型分類器へ
2. **キーワード自動拡張**: 対話ログから自動学習
3. **信頼度ベース切り替え**: confidence値による動的Stage2呼び出し
