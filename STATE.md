# STATE

> このファイルを編集するのは **Supervisor だけ**です。WI のレビューが終わったら、他の作業より先にここを更新してください。
> 到着した Supervisor は、このファイルを読んだあと `git log --oneline -15` を実行し、`last_commit` より新しいコミットがあれば、まずここを実態に合わせます。
> **150 行を超えないようにします。** 古い Review log は `docs/plan/review-log.md` に移します（短く書き直して消すことはしません）。

```yaml
last_updated: 2026-10-19
last_commit: 69bcbe3          # WI-M3-01a
current_milestone: M3
current_wi: WI-M3-02
branch: improve/m1-correctness
```

## 次の 3 アクション

1. WI-M3-02: モデルの取得をコンテナで行う（Work Order 作成済み。設計を変更し、`models-init` サービスは作らず `compose run` で取得する）
2. WI-M3-03: `./lainbow` CLI（Critical: 既存の `.env` を上書きしない。Supervisor がテストを先に書く）
3. WI-M3-04 → WI-M3-05

M1 と M2 はすべて完了（M2 は 2026-10-19）。

## 決定事項（2026-10-18 にユーザーが回答）

| ID | 決定 | 反映先 |
|---|---|---|
| Q1 | テキスト埋め込みを inference サーバーに移す。single モードでのモデル入れ替え対策を含める | 計画書 D-A, WI-M2-02 |
| Q2 | イメージを GHCR で公開する | WI-M4-02 |
| Q3 / Q5 | **CPU 構成と GPU 構成でモデルの構成は変えない**（CPU は時間がかかるだけ）。既定は、類似曲検索が `muq`、自然言語検索が `muq_mulan`（現行の既定と同じ）。MuQ-MuLan は音楽とテキストの対応づけ用なので、類似曲検索には使わない | 計画書 D-F, WI-M3-04 |
| Q4 | CLAP / MERT は選択肢として残し、既定には含めない | 計画書 D-F, WI-M2-03, WI-M3-02, WI-M3-04 |

## 未決事項（ユーザーが判断するもの）

なし。

## 進行中

なし。

## ユーザーによる実機確認が必要なもの

- （M2-01）ROCm は Supervisor がこのホストの AMD Radeon 890M で確認済み（torch 2.9.1+rocm6.4、`is_available()` が True）。**CUDA は NVIDIA の GPU がないため未確認**。NVIDIA 環境で `scripts/accel-check.sh lainbow-dev/inference-cuda:<tag>` を実行して確認してもらう必要がある
- （M3-01 完了後）既存のデータディレクトリを使ったまま新しい compose で起動でき、Sola MPD の検索と類似曲検索が従来どおり動くこと

## ツールについての確認結果（WI-M0-01、2026-10-18）

- `agy --mode plan` は読み取り専用として機能する。ファイル作成を指示しても、計画を agy 自身のディレクトリに書くだけで、リポジトリは変更されない。`scout.sh` はさらにツリーの変化も検査している
- `scout.sh`（gemini-3.8-flash-low）の応答は 1 問あたり約 15 秒で、file:line 付きの回答は正確だった
- `dispatch-implementer.sh` を print モードで実行すると、権限の確認なしでファイルを編集できる（probe は約 25 秒で完了し、HEAD は動かなかった）
- agy は `--print-timeout` に達しても exit 0 を返し、ログに `print timeout after` を書く。スクリプトはこれを検出して exit 1 にする（1s で実際に発生させて確認済み）
- このマシンには shellcheck が入っていないので、`uvx --from shellcheck-py shellcheck <files>` で実行する

## 環境についての注意

- このマシンでは本番の Lainbow が `/opt/apps/lainbow` から稼働している（コンテナ名 `lainbow_*`）。作業ツリー（`~/dev/lainbow`）とは別物で、作業ツリーの `src` の編集は本番に影響しない。**本番の DB やコンテナには触らないこと**（auto mode の classifier も本番の読み取りを拒否する）
- **作業ツリーで `docker compose` をそのまま実行しないこと。** ディレクトリ名が本番（`/opt/apps/lainbow`）と同じなので、compose のプロジェクト名がどちらも `lainbow` になり、`container_name` も固定されている。作業ツリーで compose を使う場合は必ず `-p lainbow-dev` を付ける（M3-01 でプロジェクト名を `.env` から設定できるようにする）。イメージの検証は `docker build -t lainbow-dev/...` で行う
- このホストの GPU は AMD（`/dev/kfd` あり、`nvidia-smi` なし）。本番の inference は CPU の profile で動いている。ROCm イメージの GPU 認識はこのホストで確認できる
- テスト用の Postgres `lainbow_test_pg`（127.0.0.1:55432、`--rm` 付き）を起動したままにしてある。不要になったら `docker stop lainbow_test_pg`
- 本番のスキーマは `timestamp without time zone`。`schemas.py` の `sa_type=type(DateTime(timezone=True))` はクラスを渡しているため、timezone=True が効いていない。ベースラインのリビジョンは本番の実態に合わせてある

## ゲートのベースライン

2026-10-18（WI-M0-02）時点:
- ruff check: 0（設定の `include = ["src", "tests"]` がファイルに一致しておらず、**これまで ruff は何も検査していなかった**。設定を直し、import 順の 18 件を `--fix` で修正済み）
- ruff format: 0（`tasks.py` を整形済み）
- mypy: 41 件 → WI-M0-04 で 0 件（`scripts/mypy-baseline.txt` = 0）
- pytest: 18 件すべて成功（推論テストはモデルファイルの存在チェックをモックするように修正）

## Design changes

- 2026-10-18: 計画書の初版を作成（docs/plan/improvement-plan.md）
- 2026-10-18: Q1〜Q4 の回答を反映。D-A を承認済みにし、D-F（既定モデル）を追加。M2-02 に pinned モデルを、M3-04 に `LAINBOW_ANALYZE_MODELS` を追加
- 2026-10-18: M2 の順序を変更。M2-02 を 02a（inference 側）と 02b（API 側。send_task に切り替えて worker の import をやめる）に分割し、M2-01 より先に行う
- 2026-10-18: WI-M0-03 の設計を変更（テストに SQLite を使い、`TaskResult.details` を JSON / JSONB の variant にする）
- 2026-10-18: WI-M0-04（mypy を 0 にする）を追加
- 2026-10-18: Q3 を変更。CPU と GPU でモデルの構成を揃え、既定はどちらも `muq`（類似曲検索）と `muq_mulan`（自然言語検索）とする。Q5 は解消

## Review log

| 日付 | WI | 判定 | モデル | 回数 | メモ |
|---|---|---|---|---|---|
| 2026-10-18 | WI-M0-01 | PASS | Supervisor（直接実装） | - | dispatch / scout スクリプトを作成。shellcheck で問題なし、probe で動作確認 |
| 2026-10-19 | WI-M3-01b | PASS | gemini-3.8-flash-high | 1 | compose.yaml / compose.dev.yaml に統合し、旧ファイル 4 つと down.sh を削除。Supervisor が認証情報を x-app-env で明示的に渡すよう修正（env_file だけに頼ると DB と食い違う）。`-p lainbow-dev` と別ポートで実際に起動し、全サービスが healthy、scan → analyze(acoustic) → similar が通ることを確認。モデルがない状態の検索は 503 |
| 2026-10-19 | WI-M3-01a | PASS | gemini-3.8-flash-high | 1 | `*_HOST` / `*_CONNECT_PORT` を追加。旧設定だけの場合に URL が変わらないことをテストで確認 |
| 2026-10-19 | WI-M2-01 | PASS | gemini-3.8-flash-high | 1 | DCR: ROCm の index は explicit なので `pytorch-triton-rocm` をグループに追加（承認）。Supervisor が 3 つのイメージをビルドし（cpu 1.7GB / 49 秒、cuda 7.9GB / 15 分、rocm 16.3GB / 26 分）、ROCm は GPU の認識を実機で確認。accel-check を修正（slim イメージには render グループがないため数値の GID を渡す、`set --` で第 2 引数が上書きされる不具合）。torch のバージョンを 3 つとも 2.9.1 に固定（cuda だけ 2.11 に解決されていた） |
| 2026-10-18 | WI-M2-02b | PASS | gemini-3.8-flash-high | 1 | API は send_task でタスクを投入し、torch / librosa / laion_clap / muq / mutagen を import しない（サブプロセスのテストで確認）。検索は inference に依頼し、到達できなければ 503 を返す |
| 2026-10-18 | WI-M2-02a | PASS | gemini-3.8-flash-high | 1 | `POST /api/v1/text-embedding/{model}` と pinned モデル（既定 muq_mulan）。既知の制約: ロックが 1 つなので、バッチ推論中のテキスト検索は推論の完了を待つ（M5 の候補: モデル単位のロック） |
| 2026-10-18 | WI-M2-05 | PASS | gemini-3.8-flash-high | 1 | Critical Module。Supervisor がテスト 4 本と legacy の fixture（600efa1 の create_all で生成）を先に作成。リビジョン ID は Alembic の 32 文字制限のため `0002_rename_vector_collection` に（DCR として妥当）。check.sh に postgres テストを追加 |
| 2026-10-18 | WI-M2-04 | PASS | gemini-3.8-flash-high | 1 | -502 行。pymilvus を削除し、移行スクリプトを tools/ へ移動。README の 1 文を Supervisor が言い換え |
| 2026-10-18 | WI-M1-07 | PASS | gemini-3.8-flash-high | 2 | REVISE 1 回: 全対象を 1 回の `in_()` に渡していたため、約 6.5 万曲を超えると psycopg のパラメータ上限に達する。1000 件ずつのチャンクで解消。コレクション全体を同じ統計で再 upsert する |
| 2026-10-18 | WI-M1-02 | PASS | gemini-3.8-flash-high | 2 | REVISE 1 回: full 戦略が推論エラーも曲単位で握りつぶし、推論サーバーが停止すると全曲が FAILED になっていた。読み込みと推論の失敗を分離して解消。`retry_failed` を追加（既定の挙動は変えない） |
| 2026-10-18 | WI-M2-03 | PASS | gemini-3.8-flash-high | 1 | 10 分超（Bash の上限でバックグラウンドに移行）。-774/+66 行。registry の値はテストで固定。F3 は xfail のまま |
| 2026-10-18 | WI-M1-08 | PASS | gemini-3.8-flash-high | 1 | /stats は COMPLETED のみ数える（Sola MPD に表示される件数は正しい値まで減る）。limit の範囲検証、Qdrant client の再利用、echo=False、visibility_timeout の削除、default_rng。test_audio の random のテスト 1 件を許可の範囲内で修正 |
| 2026-10-18 | WI-M1-06 | PASS | gemini-3.8-flash-high | 1 | 推論は threadpool で実行し、ModelManager は RLock で直列化。/health が推論中も応答することをテストで確認 |
| 2026-10-18 | WI-M1-05 | PASS | gemini-3.8-flash-high | 1 | リトライは接続エラーと 502/503/504 だけ。指数バックオフで最大 5 回、タイムアウトは 300 秒、client は再利用 |
| 2026-10-18 | WI-M1-04 | PASS | gemini-3.8-flash-high | 1 | UNREADABLE 状態を追加。読めないファイルは削除せずにスキップする。理由のない `type: ignore[attr-defined]` が 1 件あるが軽微なので許容 |
| 2026-10-18 | WI-M1-03 | PASS | gemini-3.8-flash-high | 1 | 音響特徴量は vectorize の段階でだけ数える。進捗は 100 で頭打ちにし、完了判定は `>=` |
| 2026-10-18 | WI-M1-01 | PASS | gemini-3.8-flash-high | 1 | 約 5 分。Supervisor が先に 5 本のテストを書いた。stale は既定 1800 秒。CPU 向けには M3-04 で 7200 秒にする |
| 2026-10-18 | WI-M0-03 | PASS | gemini-3.8-flash-high | 1 | 約 8 分。SQLite のハーネス、21 passed / 2 xfailed（F2, F3）。本番コードの変更は許可した 2 点のみ（JSON variant。Postgres の DDL が JSONB のままであることを確認済み。`UUID(task_id)` での比較） |
| 2026-10-18 | WI-M0-04 | PASS | gemini-3.8-flash-high | 1 | 約 9 分。型注釈のみ。挙動の差分は 2 点（Qdrant の delete に明示的な PointIdsList を渡す、query_vector が None のベクトルで None を返す）で、どちらも無害 |
| 2026-10-18 | WI-M0-02 | PASS | Supervisor（直接実装） | - | check.sh、ruff 設定の修正、pre-commit の整理、推論テストの修正。M0-04 を追加 |
