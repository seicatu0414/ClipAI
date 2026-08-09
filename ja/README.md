# ClipAI コンテキストパック

ClipAIは、個人向けAI編集アシスタントのためのローカルファーストな研究開発プロジェクトです。

## 問題

同時視聴者数が概ね3～10人の小規模配信者には、専任の切り抜き編集者がおらず、長時間のアーカイブを確認できないため、面白い場面が発見されないままになることがよくあります。

## プロダクト目標

長時間のアーカイブを、人が確認・編集できる量の順位付きハイライト候補へ絞り込みます。代表的な成功目標は、4時間の配信を約20～30件の有用な候補に減らすことです。

## 現在の範囲

- プラットフォーム：YouTube
- 入力：ライブ配信アーカイブと通常投稿動画
- 出力：順位と説明を備えた切り抜き候補
- 最終編集：人
- 開発環境：ローカルマシン
- 対象GPU：NVIDIA RTX 4070 Ti

## 文書案内

- [`docs/product.md`](docs/product.md)：利用者、範囲、要件、対象外
- [`docs/architecture.md`](docs/architecture.md)：システム境界とAIパイプライン
- [`docs/domain.md`](docs/domain.md)：中核概念と所有関係
- [`docs/roadmap.md`](docs/roadmap.md)：段階的な提供順序
- [`docs/experiments.md`](docs/experiments.md)：仮説、データセット、評価
- [`docs/decisions.md`](docs/decisions.md)：合意済みの製品・技術判断

英語ツリーがAIエージェント向けの正本で、日本語ツリーはオーナーレビュー用に同じ構造と意味を維持します。

## ローカル開発

`.env.example`を`.env`へコピーし、`docker compose up --build`を実行します。APIヘルスエンドポイントは `http://localhost:8000/health` です。APIとWorkerは別プロセスで、重い処理はWorkerだけが担当します。終了は `docker compose down` です。

## v0.1 ローカル文字起こし

ローカル動画は`data/`へ配置すると、コンテナ内の`/data`として利用できます。CPUでも安全に動く構成は`docker compose up --build`、NVIDIA GPUを使う構成は`docker compose -f compose.yaml -f compose.gpu.yaml up --build`で起動します。

`POST /v1/transcription-jobs`へ`{"source":"/data/example.mp4"}`またはYouTube動画URLを送信します。`GET /v1/transcription-jobs/{job_id}`で進捗と失敗内容を確認し、完了後は`GET /v1/transcripts/{transcript_id}/segments?offset=0&limit=100`で時刻順の区間をページ単位に取得します。

## v0.2 基本イベント検出

完了した文字起こしIDを`POST /v1/event-detection-jobs`へ送信し、`GET /v1/event-detection-jobs/{job_id}`で進捗を確認します。完了後は`GET /v1/transcripts/{transcript_id}/events`で、種類、開始・終了時刻、信頼度、根拠信号、説明を持つタイムラインを取得できます。

検出はRMS音量・無音特徴とバージョン付き日本語文字起こしルールを使用し、閾値は`CLIPAI_EVENT_*`環境変数で変更できます。Eventは証拠区間であり、順位付きClipCandidateではありません。
