# AGENTS.md — Lainbow で作業するすべてのエージェントへ

このファイルは、Lainbow リポジトリで作業するすべてのエージェント（Claude Code、agy、その他）が最初に読むものです。
改善計画の本体は [docs/plan/improvement-plan.md](docs/plan/improvement-plan.md)、進捗は [STATE.md](STATE.md) にあります。

---

## 1. 役割 — セッション開始時にどちらかを決める

トークンを節約するため、作業を 2 つの役割に分けます（tradr プロジェクトと同じ方式）。

| | **Supervisor** | **Implementer** |
|---|---|---|
| モデル | 高性能モデル（Claude Code 上の Claude Opus など） | 安価・高速なモデル（`agy` 経由の Gemini Flash） |
| やること | 計画、Work Order の作成、レビュー、ゲート実行、コミット、STATE.md の更新 | 渡された Work Order を 1 件だけ実装する |
| `docs/`・`AGENTS.md`・`STATE.md` の編集 | する | **しない** |
| コミット | する（PASS 後のみ） | **しない** |
| 実装コードを書く | 原則として書かない（例外は §5 の Critical Module のテストと M0） | 書く |

**判断に迷ったら Supervisor として振る舞ってください。** Implementer になるのは、プロンプトとして Work Order が渡された場合だけです。

---

## 2. Supervisor の手順

### 2-1. 到着時に読む順番

```
1. STATE.md                        現在地、進行中の WI、次の 3 アクション、未決事項
2. docs/plan/improvement-plan.md   該当するマイルストーンと WI の節だけ読む（全文は不要）
3. git log --oneline -15           STATE.md の last_updated より後に何が起きたか
4. git status                      Implementer の作業が未レビューのまま残っていないか
```

STATE.md の `last_commit` より新しいコミットがあれば、まず STATE.md を実態に合わせます。

### 2-1b. 開発環境とゲート

```sh
uv sync --group dev --group api --group batch --group inference   # CPU 版 torch。GPU は不要
scripts/check.sh   # ruff check / ruff format --check / mypy（ベースライン以下）/ pytest / shellcheck
```

mypy は `scripts/mypy-baseline.txt` の件数を超えたら失敗します。件数が減ったらその値を下げ、上げることはしません。

### 2-2. 1 つの WI を回すサイクル

```
1. 計画書の WI を読み、Work Order を .agent/work-orders/<WI-ID>.md に書く
   （テンプレート: docs/plan/work-order-template.md）
2. scripts/dispatch-implementer.sh .agent/work-orders/<WI-ID>.md で Implementer に渡す
3. レビュー（§3 の手順でトークンを節約する）
4. 判定: PASS / REVISE / REDESIGN
   - PASS     → STATE.md を更新し、実装と同じコミットに含める（1 WI = 1 コミット）
   - REVISE   → 指摘だけを書いた「REVISE round N」の Work Order を再発行する
   - REDESIGN → 計画書を直し、WI を切り直す（Supervisor が自分で実装して埋めない）
5. STATE.md の Review log に 1 行追記する
```

### 2-3. 絶対ルール

- **STATE.md に書いていない進捗は、存在しないものとして扱います。** ユーザーへ報告する前に、まず STATE.md に書きます。
- **設計変更は先に docs を直してから**実装します。順番を逆にしてはいけません。
- コミットは PASS した WI だけです。push と PR 作成は**ユーザーの確認を得てから**行います。
- `--no-verify` は使いません。

---

## 3. トークン節約プロトコル（Supervisor 向け）

高価なコンテキストは「判断」に使い、「読むこと」と「書くこと」は安価なモデルに任せます。

### 3-1. 実装の委譲: `scripts/dispatch-implementer.sh`

```sh
scripts/dispatch-implementer.sh .agent/work-orders/WI-M1-01.md
# env: LAINBOW_IMPLEMENTER_MODEL (既定 gemini-3.8-flash-high)
#      LAINBOW_PRINT_TIMEOUT     (既定 30m)
```

中身は `agy --model <model> --print-timeout=<t> --add-dir <repo> --print="$(cat <order>)"` です。tradr の `ci/dispatch-implementer.sh` で得た教訓を、次の 3 点として組み込んでいます。

1. **agy の `--print-timeout` は既定値だと短すぎます。** そのため 30m を明示的に渡します。
2. **agy はタイムアウトしても exit 0 を返します。** ログに `timeout waiting for response|print timeout after` があれば失敗として扱います。
3. **バックグラウンドタスク待ちで打ち切られた場合も exit 0 です。** ログに `terminating .* background task` があれば失敗として扱います。ツリーに変化がなければ再ディスパッチし、変化があればツリーをレビューしてゲートを手元で全部実行します。

加えて、実行の前後で HEAD を比較します。**Implementer がコミットしていたら失敗**にします。

### 3-2. 調査の委譲: `scripts/scout.sh`

「X はどこで使われているか」「このファイルは何をしているか」といった読むだけの調査は、自分でファイルを開く前に Flash に要約させます。

```sh
scripts/scout.sh "src/worker/tasks.py の analyze_song_task が呼ぶ Celery canvas の構造を 15 行以内で"
# 既定モデル gemini-3.8-flash-low。回答は短く、根拠として file:line を必ず付けさせる
```

scout の回答は**仮説として扱います**。判断の根拠に使う箇所だけ、自分で `file:line` をピンポイントで読んで確認してください。

### 3-3. レビューの読み方

1. まず `git status --short` と `git diff --stat` を見る（ファイル数と行数が Work Order の範囲内か）
2. 次にゲートを自分で実行する（`scripts/check.sh`）。Implementer の「通りました」という報告は信用しない
3. 最後に、**変更されたハンクだけ**を `git diff -- <file>` で読む。変更されていないファイルを読み直さない
4. 指摘は REVISE の Work Order に「場所・何が問題か・期待する挙動」だけを書く。修正コードは書かない

### 3-4. エスカレーション

| 状況 | 対応 |
|---|---|
| 同じ WI で REVISE が 2 回続いた | `LAINBOW_IMPLEMENTER_MODEL=gemini-4-argon-medium`（または `claude-sonnet-5-5-medium`）で再ディスパッチする |
| それでも通らない | Work Order か計画の書き方を疑い、WI を小さく切り直す（REDESIGN） |
| 1 文字のタイポなどの自明な修正 | Supervisor が直接直してよい（その旨を Review log に残す） |

### 3-5. Work Order の書き方（品質の大半はここで決まります）

- **自己完結させる**: Implementer にはコンテキストがありません。触るファイル、期待する挙動、参照すべき計画書の節、範囲外のこと、ゲートを明記します。
- **大きさ**: 差分は約 300 行以内、ファイルは 6 個以内。超えるなら WI を分けます。
- **報告形式を指定する**: 変更したファイル、各ゲートの出力の末尾をそのまま貼ること、未実行のゲートは `NOT RUN` と書くこと。

---

## 4. Implementer のルール（agy で実行されるエージェントへ）

1. **Work Order の Definition of Done に書かれていることだけを行います。** 「ついでの改善」はしません。
2. `docs/`、`AGENTS.md`、`CLAUDE.md`、`STATE.md`、`.agent/` は**編集しません**。
3. **`git commit` / `push` / `checkout` / `reset` / `stash` / ブランチ操作はしません。** 変更は作業ツリーに残してください。
4. 設計上の問題を見つけたら、勝手に設計を変えず、作業を止めて報告に **Design Change Request** として書きます。
5. コード中のコメントは英語で書き、周囲のコードのスタイルに合わせます。
6. ゲートの結果は、実行したコマンドの出力をそのまま報告します。実行していないものは `NOT RUN` と書きます。

---

## 5. 不変条件（すべての WI が守るもの）

| ID | 内容 | 理由 |
|---|---|---|
| I1 | **Sola MPD が使う API の互換性を保つ**: `GET /health`, `GET /api/v1/stats`, `GET /api/v1/songs/search`, `GET /api/v1/songs/{path}/similar`, `GET /api/v1/songs/{path}/analysis`, `POST /api/v1/batch/{scan,vacuum,analyze}/run`, `GET /api/v1/batch/tasks/{id}` のパス・パラメータ・レスポンスのフィールド | `sola_mpd/packages/backend/src/advanced_search/services/AdvancedSearchApiHttp.ts` が依存している |
| I2 | **既存環境のデータを壊さない**: Postgres のデータと Qdrant のコレクション名を保持する。マイグレーションは非破壊にする | 作者の本番環境がすでに動いている |
| I3 | **Qdrant の point ID は `uuid5(QDRANT_ID_NAMESPACE, file_path)` のまま** | 変えると既存ベクトルとの対応がすべて失われる |
| I4 | `file_path` は音楽ルートからの相対パス（MPD の `music_directory` 基準と同じ） | Sola MPD 側がこのパスでクエリしている |
| I5 | ゲート（`scripts/check.sh`）は GPU も実行中の Docker スタックも無しで通る | Implementer と CI の環境で再現できるようにするため |

**Critical Module**（Supervisor が先にテストを書き、Implementer はそのテストを通す実装を書く）:
Alembic のベースライン検出と stamp（I2）、Qdrant の point ID（I3）、`./lainbow setup` が既存の `.env` を上書きしないこと（I2）。

---

## 6. コミット規約

- 1 WI = 1 コミット。メッセージは既存の履歴に合わせて `fix:` / `feat:` / `chore:` / `docs:` で始め、本文に `WI-ID` を書きます。
- コミットには STATE.md の更新を必ず含めます。
- 作業ブランチはマイルストーンごとに `improve/m<N>-<topic>` とし、main へは PR でマージします（PR の作成はユーザーの確認を得てから）。
