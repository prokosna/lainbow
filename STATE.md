# STATE

> このファイルを編集するのは **Supervisor だけ**です。WI のレビューが終わったら、他の作業より先にここを更新してください。
> 到着した Supervisor は、このファイルを読んだあと `git log --oneline -15` を実行し、`last_commit` より新しいコミットがあれば、まずここを実態に合わせます。
> **150 行を超えないようにします。** 古い Review log は `docs/plan/review-log.md` に移します（短く書き直して消すことはしません）。

```yaml
last_updated: 2026-10-18
last_commit: 281e946          # WI-M1-08
current_milestone: M1
current_wi: WI-M1-02
branch: improve/m1-correctness
```

## 次の 3 アクション

1. WI-M1-02（部分失敗）、WI-M1-07（正規化）を registry の上で行う
2. M2-01（アクセラレータ別の torch）、M2-04（Milvus の削除）、M2-05（Alembic）

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

- （M2-01 完了後）CUDA と ROCm のコンテナで `scripts/accel-check.sh` が `True` を返すこと
- （M3-01 完了後）既存のデータディレクトリを使ったまま新しい compose で起動でき、Sola MPD の検索と類似曲検索が従来どおり動くこと

## ツールについての確認結果（WI-M0-01、2026-10-18）

- `agy --mode plan` は読み取り専用として機能する。ファイル作成を指示しても、計画を agy 自身のディレクトリに書くだけで、リポジトリは変更されない。`scout.sh` はさらにツリーの変化も検査している
- `scout.sh`（gemini-3.8-flash-low）の応答は 1 問あたり約 15 秒で、file:line 付きの回答は正確だった
- `dispatch-implementer.sh` を print モードで実行すると、権限の確認なしでファイルを編集できる（probe は約 25 秒で完了し、HEAD は動かなかった）
- agy は `--print-timeout` に達しても exit 0 を返し、ログに `print timeout after` を書く。スクリプトはこれを検出して exit 1 にする（1s で実際に発生させて確認済み）
- このマシンには shellcheck が入っていないので、`uvx --from shellcheck-py shellcheck <files>` で実行する

## ゲートのベースライン

2026-10-18（WI-M0-02）時点:
- ruff check: 0（設定の `include = ["src", "tests"]` がファイルに一致しておらず、**これまで ruff は何も検査していなかった**。設定を直し、import 順の 18 件を `--fix` で修正済み）
- ruff format: 0（`tasks.py` を整形済み）
- mypy: 41 件 → WI-M0-04 で 0 件（`scripts/mypy-baseline.txt` = 0）
- pytest: 18 件すべて成功（推論テストはモデルファイルの存在チェックをモックするように修正）

## Design changes

- 2026-10-18: 計画書の初版を作成（docs/plan/improvement-plan.md）
- 2026-10-18: Q1〜Q4 の回答を反映。D-A を承認済みにし、D-F（既定モデル）を追加。M2-02 に pinned モデルを、M3-04 に `LAINBOW_ANALYZE_MODELS` を追加
- 2026-10-18: WI-M0-03 の設計を変更（テストに SQLite を使い、`TaskResult.details` を JSON / JSONB の variant にする）
- 2026-10-18: WI-M0-04（mypy を 0 にする）を追加
- 2026-10-18: Q3 を変更。CPU と GPU でモデルの構成を揃え、既定はどちらも `muq`（類似曲検索）と `muq_mulan`（自然言語検索）とする。Q5 は解消

## Review log

| 日付 | WI | 判定 | モデル | 回数 | メモ |
|---|---|---|---|---|---|
| 2026-10-18 | WI-M0-01 | PASS | Supervisor（直接実装） | - | dispatch / scout スクリプトを作成。shellcheck で問題なし、probe で動作確認 |
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
