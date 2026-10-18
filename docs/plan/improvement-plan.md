# Lainbow 改善計画

- 作成日: 2026-10-18（コードベース全体のレビュー結果をもとに作成）
- 読む人: すべてのエージェント。進め方は [AGENTS.md](../../AGENTS.md)、進捗は [STATE.md](../../STATE.md) を参照してください。
- この文書を編集するのは Supervisor だけです。設計を変えるときは、先にこの文書を直してから実装します。

---

## 1. 目的

1. **正しく動くこと**: レビューで見つかった不具合（タスクがスタックする、バッチの一部失敗が全体に波及する、GPU が使われない、など）を直す。
2. **導入しやすいこと**: ユーザーが自分の環境（NVIDIA / AMD / CPU のみ、1 台構成 / 分散構成）で、**`git clone` のあと数コマンドで動かせる**ようにする。
3. **保守しやすいこと**: 重複を減らし、テストと CI で品質を機械的に守る。安価なモデルでも安全に実装できる構造にする。

### 導入体験の目標（M3 完了時点）

```sh
git clone https://github.com/prokosna/lainbow && cd lainbow
./lainbow setup      # 環境を検出 → 対話で .env を生成 → モデルを取得
./lainbow up         # 起動し、ヘルスチェックが通るまで待つ
./lainbow scan && ./lainbow analyze
```

- GPU の種類は自動で検出します（`nvidia-smi` と Docker の nvidia runtime、`/dev/kfd`、どちらも無ければ CPU）。
- Dockerfile の `FROM` をユーザーが書き換える必要はありません。ホストに Python や uv を入れる必要もありません（必要なのは Docker と Compose v2 だけ）。

---

## 2. 現状の問題

### 2-1. レビューで見つかった不具合（2026-10-18）

| # | 重要度 | 内容 | 場所 | 対応する WI |
|---|---|---|---|---|
| F1 | 🔴 | CUDA/ROCm イメージでも `uv.lock` の torch が `+cpu` のみなので、GPU が使われない | `pyproject.toml:84-87`, `docker/inference*.Dockerfile` | M2-01 |
| F2 | 🔴 | PENDING / RUNNING のまま残ったタスクが、同名タスクを永久に SKIPPED にする（analyze の例外時に `status=RUNNING` のまま、`record_progress` が try の外、`.delay()` 失敗時に PENDING が残る、クラッシュ） | `tasks.py:38-52,516,759,773`, `batch.py:26-28` | M1-01 |
| F3 | 🔴 | 埋め込みのバッチで 1 曲失敗すると KeyError でチャンク全体が失敗し、毎回やり直しになる | `tasks.py:624,649,674,699` | M1-02 |
| F4 | 🔴 | 音響特徴量の進捗を二重に数えるため、途中で SUCCESS になり、その後 100% を超える | `tasks.py:535,588` | M1-03 |
| F5 | 🟠 | mutagen の例外が 1 件あるだけで Scan / Vacuum 全体が失敗する | `audio.py:96-115`, `tasks.py:111,315` | M1-04 |
| F6 | 🟠 | 推論クライアントがすべての例外を 10 回 × 30 秒リトライし、タイムアウトも 30 秒しかない | `inference_client.py:14-18` | M1-05 |
| F7 | 🟠 | 推論エンドポイントが `async def` のまま同期推論を実行しており、`/health` まで止まる | `inference/v1/endpoints/inference.py:17` | M1-06 |
| F8 | 🟠 | API 側の `TextEmbeddingService` がスレッドセーフでない | `text_embedding.py:83-98` | M2-02 で解消（API からモデルを除く） |
| F9 | 🟠 | 音響特徴量の正規化に使う統計値が実行ごとに変わり、古いベクトルと比較できなくなる | `tasks.py:550` | M1-07 |
| F10 | 🟠 | `/stats` が状態を見ずに件数を数えている | `stats.py:34-62` | M1-08 |
| F11 | 🟠 | `limit` の値を検証していない | `songs.py:147,220` | M1-08 |
| F12 | 🟡 | Qdrant クライアントを呼び出しごとに生成している、`echo=True`、`print`、`visibility_timeout` が無効、起動時の空の警告、`np.random.seed` によるグローバル汚染、本番設定でも `--reload` | 各所 | M1-08, M3-01, M5 |
| F13 | 🔵 | Milvus 時代の残骸（設定、カラム名、依存、コメントアウトされた compose） | 各所 | M2-04 |
| F14 | 🔵 | `clap`/`mert`/`muq`/`muq_mulan` と `generate_song_embedding` の大量の重複 | `src/worker/*` | M2-03 |
| F15 | 🔵 | マイグレーションの仕組みがない（`create_all` のみ） | `db_manager.py` | M2-05 |
| F16 | 🔵 | テストがほぼ無い。pre-commit が存在しないファイルを除外していて、mypy の依存も古い | `tests/`, `.pre-commit-config.yaml` | M0-02, M0-03 |

### 2-2. 導入しにくい原因

| # | 内容 |
|---|---|
| S1 | compose ファイルが 4 つあり、個別に `up` する必要がある。1 台構成用の入口がない |
| S2 | モデルはホストで `uv run --with ...` を使ってダウンロードする必要がある（ホストに uv と Python が必要） |
| S3 | README に「GPU に合わせて Dockerfile の `FROM` を書き換えてください」とある。しかも F1 があるので書き換えても GPU は使われない |
| S4 | README の `--scale batch-cpu=N` が実際のサービス名 `batch_cpu` と一致しておらず、コマンドが失敗する |
| S5 | DB の認証情報が compose にハードコードされており、`.env` で変えても反映されない |
| S6 | API イメージに torch、laion-clap、muq、librosa が入っていて巨大。API 起動時に MuQ-MuLan をロードするため、API 側にも `models/` が必要 |
| S7 | イメージを毎回ローカルでビルドする必要がある（配布されたイメージがない） |
| S8 | すべてのサービスが `host.docker.internal` 経由で通信する（1 台構成でもホストのポートを経由する） |
| S9 | `src/` を bind mount し `--reload` を付けた開発用の設定が、そのまま本番の既定になっている |
| S10 | CPU だけの環境向けの既定値（チャンクサイズ、タイムアウト、モデル選択）や説明がない |
| S11 | Sola MPD と接続する際の注意（音楽ディレクトリのパスを MPD の `music_directory` と揃える必要がある）が書かれていない |

---

## 3. 目標アーキテクチャ（M2 から M3 完了時点）

```
compose.yaml（1 ファイル。profile で切り替える）
├─ postgres, rabbitmq, qdrant            … 常に起動。認証情報は .env から読む
├─ api        (python:3.12-slim, 軽量)   … torch なし。テキスト埋め込みは inference に HTTP で依頼する
├─ batch      (python:3.12-slim + ffmpeg)… Celery ワーカー
├─ inference-{cpu|cuda|rocm}  profile    … アクセラレータごとに torch のビルドを分ける
├─ models-init (one-shot)                … 推論イメージで必要なモデルだけを ./models にダウンロードする
└─ flower      profile: debug

compose.dev.yaml   … src の bind mount と --reload（開発用）
./lainbow          … bash で書いたセットアップと運用の CLI
```

**設計上の判断**（変更する場合は先にこの節を直すこと）

- **D-A: テキスト埋め込みを inference サーバーに移す（2026-10-18 承認、Q1）。** API から torch とモデルを除き、F8 と S6 をまとめて解消します。代償として、`/songs/search` だけは inference サーバーが動いていることが前提になります（`/similar`、`/stats`、`/batch/*` は影響を受けません）。single モードで analyze と検索が重なったときにモデルの入れ替えが繰り返されないよう、テキスト用モデルを常駐させる仕組みを WI-M2-02 に含めます。
- **D-F: 既定のモデル（2026-10-18 決定、Q3・Q4・Q5）。** CPU 構成と GPU 構成でモデルの構成は変えません（CPU は時間がかかるだけです）。既定は、類似曲検索用の `muq` と、自然言語検索用の `muq_mulan` です（現在の既定と同じ）。MuQ-MuLan は音楽とテキストの対応づけ用のモデルなので、類似曲検索には使いません。CLAP と MERT は選択肢として残しますが、既定には含めず、ダウンロードもしません。アクセラレータによって変えてよいのは、チャンクサイズやタイムアウトなどの**性能面の設定だけ**です。埋め込みの結果が変わる設定（`MUQ_FILE_DURATION_SEC` など）は、アクセラレータによって変えません。Sola MPD は analyze をリクエスト本文なしで呼ぶため、既定値はサーバー側の設定 `LAINBOW_ANALYZE_MODELS` から決めます（I1 の範囲内です。本文に `models` を指定すれば従来どおり上書きできます）。
- **D-B: GPU 用イメージのベースも `python:3.12-slim` にする。** PyTorch の cu128 / rocm wheel は CUDA / ROCm のランタイムを同梱しているので、ホストに必要なのはドライバと container toolkit だけです。巨大なベースイメージに依存しなくなります。
- **D-C: 1 台構成ではサービス名で通信する。** 分散構成の場合だけ、`.env` の `DATABASE_SERVER_ENDPOINT` / `INFERENCE_API_URL` で上書きします（既存の変数名は互換のため維持）。
- **D-D: CLI は bash で書く。** ホストに Python を要求しないためです。品質は shellcheck で担保します。
- **D-E: `DATABASE_DATA_ROOT_DIR/{postgres_data,qdrant_data,rabbitmq_data}` の配置は変えない。** 既存ユーザーがそのまま移行できるようにするためです（I2）。

---

## 4. マイルストーンと Work Item

各 WI は「目的 / 変更対象 / 完了条件（DoD）/ ゲート」の形で書きます。Supervisor はこれを Work Order に具体化します。
**共通ゲート**（M0-02 以降）は `scripts/check.sh` です（ruff check、ruff format --check、mypy、pytest）。WI ごとの追加ゲートは個別に記載します。

### M0: エージェント作業の基盤（Supervisor が直接実装する）

**WI-M0-01 dispatch / scout スクリプト**
- 変更対象: `scripts/dispatch-implementer.sh`, `scripts/scout.sh`, `.gitignore`
- DoD:
  - `dispatch-implementer.sh` は tradr の `ci/dispatch-implementer.sh` を移植したもの。環境変数は `LAINBOW_IMPLEMENTER_MODEL`（既定 `gemini-3.8-flash-high`）と `LAINBOW_PRINT_TIMEOUT`（既定 `30m`）。タイムアウトの検出、バックグラウンド打ち切りの検出、Implementer のコミット検出は AGENTS.md §3-1 のとおりに実装する。
  - `scout.sh "<question>"` は `agy --model ${LAINBOW_SCOUT_MODEL:-gemini-3.8-flash-low}` で読み取り専用の質問をする。プロンプトの先頭に「ファイルを編集しない、回答は簡潔に、根拠を file:line で示す」を付ける。読み取り専用を保証するフラグ（`--mode plan` など）が効くかを初回に確認し、結果を STATE.md に書く。
  - `.gitignore` に `.agent/` と `.dispatch-*.log` を追加する。
- ゲート: `shellcheck scripts/*.sh`。ダミーの Work Order（README を 1 行要約させるだけ）を実際にディスパッチし、ログが出ることと HEAD が動かないことを確認する。

**WI-M0-02 開発環境とゲート**
- 変更対象: `pyproject.toml`（`dev` グループ）、`scripts/check.sh`、`.pre-commit-config.yaml`
- DoD:
  - GPU なしで `uv sync --group dev --group batch` を実行すればゲートが回る状態にする（必要なら CPU 版 torch を含む `test` グループを作る）。
  - `check.sh` は 4 つのゲートを、途中で失敗しても全部実行し、最後にまとめて終了コードを返す。
  - pre-commit から存在しない `test_embeddings.py` の除外と、古い mypy 依存（redis、pymilvus）を外す。
  - 現時点の ruff / mypy の違反数を**ベースライン**として STATE.md に記録する。直せるものは直し、残ったものは M1 以降で「増やさない」ことをゲートにする。
- ゲート: `scripts/check.sh` が実行でき、ベースラインが記録されていること。

**WI-M0-03 タスクロジックのテスト基盤**
- 変更対象: `src/domain/schemas.py`（1 行）, `tests/conftest.py`, `tests/worker/test_tasks.py`
- 設計（2026-10-18 に変更。当初の「SQLite を使わない」は撤回）:
  - DB はインメモリの SQLite（StaticPool）を使う。`TaskResult.details` を `JSON().with_variant(JSONB(), "postgresql")` に変える。Postgres の DDL は JSONB のままなので I2 に反しない。
  - `domain.session_manager.SessionFactory` を monkeypatch で差し替え、`db_session_context` をそのまま使う。
  - Qdrant は `domain.vector_store_utils` の関数を、メモリ上の偽実装に差し替える。推論は各ワーカーモジュールの `run_inference` を、次元の合った決定的なベクトルを返す偽実装に差し替える。
  - 音声は tmp_path に生成した短いサイン波の WAV を使い、`domain.config.MUSIC_NAS_ROOT_DIR` を差し替える。
  - Celery のタスクは関数として同期的に呼ぶ。`chord` などの canvas はモックする。
  - Postgres が必要なテストは `@pytest.mark.postgres` を付け、既定ではスキップする。
- DoD: 現在の正しい挙動を固定するテスト（スキップ判定、record_progress の加算と完了、埋め込み生成の成功）と、既知の不具合の再現テスト（F2: 古い RUNNING が新しいタスクをブロックする、F3: 1 曲の失敗でチャンク全体が失敗する）を書く。再現テストは `xfail(strict=True)` にし、M1 の各 WI で xfail を外す。
- 備考: M1 の各 WI は、このフィクスチャを使って**再現テストを先に書く**。

**WI-M0-04 mypy のエラーを 0 にする**（M0-02 で追加。最初に Implementer へ渡す WI）
- 変更対象: `src/` 全体（型注釈のみ。挙動は変えない）
- DoD: `uv run mypy src` のエラーを 0 にし、`scripts/mypy-baseline.txt` を 0 にする。`# type: ignore` を新たに追加するときは、エラーコードと理由を書いたものに限る。ロジックの変更はしない。
- 備考: 内訳（2026-10-18 時点で 41 件）は type-arg 26、unused-ignore 5、arg-type 4、assignment 2、その他 4。

### M1: 正しさの修正（レビュー結果への対応）

**WI-M1-01 スタックしたタスクの回復（F2）** — Supervisor がテストを先に書く
- 変更対象: `src/worker/tasks.py`, `src/domain/task_manager.py`, `src/api/v1/endpoints/batch.py`, `src/domain/config.py`
- DoD:
  - `_skip_task_if_already_running` は、`updated_at` が `TASK_STALE_AFTER_SEC`（既定 1800）より古い PENDING / RUNNING を FAILURE（メッセージは "stale"）にしてから判定する。
  - `analyze_song_task` 本体の例外時は FAILURE にする（RUNNING のまま記録しない）。`_handle_exception` の RUNNING 指定はサブタスクの場合だけにする。
  - `generate_song_embedding` の `record_progress` を try の中に入れ、すべてのサブタスクが例外を外に出さないようにする（chord を必ず完了させるため）。
  - API で `.delay()` が失敗した場合、そのタスクを FAILURE にしてから 500 を返す。
- テスト: 上記 4 つの経路それぞれに再現テストを書く。

**WI-M1-02 埋め込みバッチの部分失敗（F3）**
- 変更対象: `src/worker/tasks.py`
- DoD:
  - `embeddings_dict` にあるパスだけを upsert し、COMPLETED として記録する。結果に無いパスは `SongEmbedding(status=FAILED)` として記録する。
  - 次回の analyze 対象の選び方は従来どおり「レコードが無いもの」とする。FAILED を再試行するためのオプション `retry_failed: bool = False` を `RunAnalysisRequest` に追加する（I1 の範囲内で、既定の挙動は変えない）。
  - 失敗した曲も `processed_count` に数え、全体の進捗が 100% に到達できるようにする。
- 備考: M2-03 の後で実装すると差分が小さくなるため、M2-03 を先に行ってもよい（Supervisor が判断する）。

**WI-M1-03 進捗の二重計上（F4）**
- DoD: 音響特徴量は vectorize の段階でだけ数える（extract の段階では `record_progress` を呼ばない）。`total_count` の意味（曲 × モデル）を docstring に書く。進捗が 100% を超えないことをテストで確認する。

**WI-M1-04 Scan / Vacuum の耐障害性（F5）**
- 変更対象: `src/worker/audio.py`, `src/worker/tasks.py`
- DoD: `extract_metadata` と `check_audio_file_status` はファイル単位で例外を捕捉し、ログを出して None または OK を返す（壊れたファイルを「存在しない」として扱って削除しない）。スキップしたファイルの数をタスクのメッセージに含める。

**WI-M1-05 推論クライアントのリトライ方針（F6）**
- 変更対象: `src/domain/inference_client.py`, `src/domain/config.py`
- DoD:
  - リトライするのは接続エラー、読み取りタイムアウト、502 / 503 / 504 だけにする。それ以外の HTTP エラーは即座に例外を投げる。
  - 待ち時間は指数バックオフ（5s → 最大 60s、最大 5 回）。`httpx.Client` はモジュールレベルで再利用する。
  - `INFERENCE_REQUEST_TIMEOUT` の既定値を 300 にする（CPU で推論する場合を考慮）。

**WI-M1-06 推論サーバーの並行実行（F7）**
- 変更対象: `src/inference/v1/endpoints/inference.py`, `src/inference/core/model_manager.py`
- DoD:
  - エンドポイントの中では、推論部分を `run_in_threadpool` で実行する。
  - `ModelManager` に `threading.Lock` を入れ、モデルの読み込み・解放・推論を直列化する（single モードの安全のため）。
  - `INFERENCE_LOAD_MODELS_ON_STARTUP` の空要素を除外する。
  - 推論の実行中も `/health` が応答することをテストで確認する。

**WI-M1-07 音響特徴量の正規化を一貫させる（F9）**
- DoD: 統計値を計算し直したら、`acoustic_features` コレクション全体を再 upsert する（55 次元なので 10 万曲でも安価）。ベクトル化は analyze 1 回につき 1 タスクにまとめ、チャンクごとに統計を計算し直すことはやめる。

**WI-M1-08 API と細部の修正（F10, F11, F12 の一部）**
- DoD:
  - `/stats` は `status == COMPLETED` の件数を数える（フィールド名は I1 に従い変えない）。
  - `limit` は `Query(ge=1, le=200)` で検証する。
  - `print` を logger に置き換える。
  - `db_manager` の engine を `echo=False` にし、session_manager の engine を再利用する。
  - Qdrant クライアントをモジュールレベルで再利用する。
  - `visibility_timeout` を削除する。
  - `np.random.seed` を `np.random.default_rng(42)` に置き換える。

### M2: 依存関係とアーキテクチャ（移植しやすいイメージにする）

**WI-M2-01 アクセラレータ別の torch（F1, S3）** — Supervisor が設計を確定してから発行する
- 変更対象: `pyproject.toml`, `uv.lock`, `docker/inference.Dockerfile`（cpu / cuda / rocm を build-arg で切り替える 1 ファイルに統合する）
- DoD:
  - uv の `[tool.uv] conflicts` とグループ別の `sources` を使い、`inference-cpu` / `inference-cuda`（cu128 index）/ `inference-rocm`（torch の lock バージョンが公開している最新の rocm index）のグループを作る。
  - Dockerfile のベースは `python:3.12-slim` とする（D-B）。`ARG ACCEL=cpu|cuda|rocm` で `uv sync --group inference-${ACCEL}` を実行する。
  - コンテナ内で `python -c "import torch;print(torch.__version__, torch.cuda.is_available())"` を実行するスクリプト `scripts/accel-check.sh` を用意する。
- ゲート: 3 種類すべてで `docker build` が成功すること。cuda / rocm で `is_available()` が True になることは**ユーザーが実機で確認する**（STATE.md の要確認事項に入れる）。

**WI-M2-02 テキスト埋め込みを inference に移す（D-A, F8, S6）** — Q1 承認済み
- 変更対象: `src/inference/v1/endpoints/`（`POST /api/v1/text-embedding/{model}` を追加。JSON で `{"text": str}` を受け取り、`{"embedding": [float]}` を返す）、`src/inference/models/{clap,muq_mulan}_model.py`（`embed_text` を追加）、`src/domain/inference_client.py`（`run_text_embedding` を追加）、`src/api/core/text_embedding.py`（削除）、`src/api/main.py`、`src/api/v1/endpoints/songs.py`、`pyproject.toml`（`api` グループから torch、laion-clap、muq、librosa、pymilvus を外す）
- DoD:
  - `/songs/search` のレスポンスが変わらないこと（I1）。API は `worker` パッケージを import しない（M2-03 の registry を使う）。
  - **モデル入れ替えへの対策**: `ModelManager` に常駐させるモデルの一覧 `INFERENCE_PINNED_MODELS`（既定 `muq_mulan`）を追加する。single モードでも、ここに入っているモデルは解放の対象から外す。モデルを入れ替えるのは pinned 以外のモデルの間だけにする。
  - inference サーバーが落ちているときの `/songs/search` は 503 を返し、メッセージで inference サーバーが必要であることを伝える（500 にしない）。
- テスト: single モードで `muq` → テキスト埋め込み（`muq_mulan`）→ `muq` の順に呼んでも、`muq_mulan` が解放されず、`muq` も再ロードされないこと。
- 順序: M2-03 の後に行う。

**WI-M2-03 埋め込みモデルの registry（F14）**
- 変更対象: `src/domain/embedding_models.py`（新規）、`src/worker/embeddings.py`（clap / mert / muq / muq_mulan を統合する）、`src/worker/tasks.py`、`src/api/v1/endpoints/songs.py`
- DoD:
  - `ModelSpec(name, collection_name, dimension, sample_rate, strategy: "segments" | "full")` の dict を 1 つ作り、コレクション名や次元はすべてここから参照する。
  - `generate_song_embedding` の 4 つの分岐を 1 つにまとめる。
  - コレクション名と次元は現状のまま変えない（I2）。
  - CLAP と MERT は registry に残し、`/similar` や analyze の本文で指定すれば使える状態を保つ（Q4: 選択肢として残す）。

**WI-M2-04 Milvus の残骸を削除する（F13）**
- DoD:
  - `MILVUS_*` の設定、`pymilvus` 依存、未使用の `id_field` / `vector_field` / `index_params` 引数、各モジュールの `INDEX_PARAMS`、compose 内のコメントアウトを削除する。
  - `migrate_milvus_to_qdrant.py` は `tools/` に移す。README には `uv run --with pymilvus --with qdrant-client tools/migrate_milvus_to_qdrant.py` と書く。
  - カラム名の変更は M2-05 で行う。

**WI-M2-05 Alembic の導入（F15）** — Critical Module: Supervisor がテストを先に書く
- DoD:
  - 現在のスキーマと一致するベースラインのリビジョンを作る。
  - API 起動時に「テーブルがあるのに `alembic_version` が無い場合はベースラインを stamp し、そのあと `upgrade head`」を実行する（既存環境 I2）。
  - 2 つ目のリビジョンで `milvus_collection_name` を `vector_collection_name` に変更する（非破壊の rename）。
  - `create_all` の呼び出しを削除する。
- テスト（`@pytest.mark.postgres`）: 空の DB、既存スキーマの DB、最新の DB の 3 状態で起動しても冪等であること。

### M3: 導入体験の改善

**WI-M3-01 compose ファイルを 1 つにまとめる（S1, S4, S5, S8, S9）**
- 変更対象: `compose.yaml`（新規）、`compose.dev.yaml`（新規）、旧 `docker-compose.*.yaml`（削除）、`down.sh`（削除）、`.env.template`
- DoD:
  - §3 の構成にする。DB の認証情報は `${POSTGRES_PASSWORD:?}` などで `.env` から必須として読む。
  - `depends_on: condition: service_healthy` を設定する。
  - batch サービスの名前を `batch` にし、並列数は `BATCH_CONCURRENCY` で指定する。
  - Flower は `profiles: [debug]` に入れる。
  - API 以外のポートは既定で `127.0.0.1` にだけ bind する。
  - データの配置は D-E のとおり維持する。
  - 分散構成のために `profiles` を `db` / `app` / `inference-*` に分け、組み合わせて起動できるようにする。
- ゲート: `docker compose config` が全 profile で通ること。CPU の profile で `up` し、全サービスが healthy になること（Supervisor が手元で確認）。

**WI-M3-02 モデルのダウンロードをコンテナで行う（S2）**
- 変更対象: `download_models.py` → `src/inference/download_models.py`（`--models muq,muq_mulan` で選択）、compose の `models-init` サービス
- DoD: 冪等にする。ダウンロードが完了したら `.complete` マーカーを書き、途中で中断されたものは再取得する。`LAINBOW_ANALYZE_MODELS`（D-F）で指定したものだけを取得する。CLAP と MERT は、利用者が明示的に指定した場合だけ取得する。

**WI-M3-03 `./lainbow` CLI（S1, S10）** — `.env` を上書きしない部分は Critical Module
- 変更対象: `lainbow`（bash）, `tests/cli/`（bats、または bash で書いたテストスクリプト）
- サブコマンド:
  - `setup`: 前提条件（docker と compose v2）のチェック、アクセラレータの検出、音楽ディレクトリの入力と存在確認、ポートとモデルの選択、ランダムな DB パスワードの生成、`.env` の生成、`models-init` の実行。`--music-dir --accel --models --yes` を付ければ非対話でも実行できる。**既存の `.env` がある場合は差分だけを追記し、上書きしない。**
  - `up` / `down` / `status`（各サービスの health と `/api/v1/stats`）/ `logs [svc]` / `scan` / `analyze [--models ...]` / `update`（pull、マイグレーション、再起動）/ `doctor`（GPU の可視性、ポートの衝突、音楽ディレクトリの権限などの診断）
- ゲート: `shellcheck lainbow`。`setup --yes` を一時ディレクトリで実行し、`.env` が期待どおりに生成されること、2 回目の実行で既存の値が保持されることをテストする。

**WI-M3-04 CPU 向けの既定値（S10）**
- DoD:
  - `AnalyzeSongTaskPayload` と `RunAnalysisRequest` の既定値を、ハードコードされたリストから `config.ANALYZE_MODELS`（環境変数 `LAINBOW_ANALYZE_MODELS`。未設定のときは `muq,muq_mulan` で、現在の挙動と同じ）に変える（D-F）。
  - `setup` はアクセラレータに関係なく `LAINBOW_ANALYZE_MODELS=muq,muq_mulan` を書き込む（D-F）。
  - `accel=cpu` の場合に変えるのは性能面の設定だけにする（`TASK_CHUNK_SIZE=2`、`INFERENCE_REQUEST_TIMEOUT=600`、`TASK_STALE_AFTER_SEC=7200`。CPU では 1 チャンクに 30 分以上かかり、M1-01 の stale 判定に誤って引っかかることがあるため）。`MUQ_FILE_DURATION_SEC` は変えない。
  - `setup` の中で、CLAP と MERT を追加するかを聞く（既定は「いいえ」）。
  - README にハードウェア別の処理時間の目安表を載せる（CPU でも同じ機能が使え、時間がかかるだけであることを明記する）。

**WI-M3-05 README を書き直す（S11 ほか）**
- 構成: Quick start（3 コマンド）/ 動作要件とハードウェアの目安 / Sola MPD との接続（API の URL、**`music_directory` と音楽ディレクトリを揃える**こと）/ 分散構成 / 旧構成からの移行（データディレクトリはそのまま使える、旧 compose で `down` してから新 compose で `up` する）/ トラブルシューティング / API リファレンス（現行の内容を維持）。

### M4: 配布と CI

**WI-M4-01 GitHub Actions による CI**: PR ごとに `scripts/check.sh`、shellcheck、`docker compose config`、CPU イメージのビルドを実行する。
**WI-M4-02 GHCR でのイメージ公開（Q2 承認済み）**: タグを打ったら `api` / `batch` / `inference-{cpu,cuda,rocm}` をビルドして push する。compose は `image: ghcr.io/prokosna/lainbow-*:${LAINBOW_VERSION}` を使い、`build:` もフォールバックとして残す。`./lainbow update` が pull を行う。
**WI-M4-03 E2E スモークテスト**: 生成したサイン波の音声を数ファイル用意し、`scan` → `analyze --models acoustic_features` → `/similar` が通ることを確認する（モデルのダウンロードは不要）。CI と `./lainbow doctor --e2e` の両方から実行できるようにする。

### M5: 堅牢化（任意）

- `LAINBOW_API_TOKEN`（任意）による Bearer 認証。Sola MPD 側の対応が必要になるので、既定では無効にする（I1）。
- RabbitMQ の `consumer_timeout` を調整する。長いタスクの `acks_late` との整合を取る。
- `ModelManager` のロックをモデル単位にする（WI-M2-02a のレビューで判明。現状は 1 つのロックなので、バッチ推論中のテキスト検索が推論の完了を待つ。pinned モデルどうしで VRAM を奪い合わない範囲で並行実行を許す）
- `extract_audio_segments` の「最後のセグメントを除外する」意図が短い曲で守られていない問題を修正する。

---

## 5. 実行順序と依存関係

```
M0-01 → M0-02 → M0-04, M0-03
          ↓
M1-01, M1-03, M1-04, M1-05, M1-06, M1-08   （互いに独立しているので順に回す）
M2-03 → M1-02, M1-07                       （registry を先に作ると差分が小さくなる）
M2-03 → M2-04 → M2-05 → M2-02a → M2-02b → M2-01   （API が torch を手放すまで、アクセラレータ別のグループは衝突する）
          ↓
M3-01 → M3-02 → M3-03 → M3-04 → M3-05
          ↓
M4-01 → M4-03 → M4-02
```

推奨する進め方: M0 → M1（M1-02 と M1-07 を除く）→ M2-03 → M1-02 / M1-07 → 残りの M2 → M3 → M4。

---

## 6. この計画の変更方法

- WI の追加・分割・削除は Supervisor がこの文書で行い、STATE.md の Design changes に 1 行で記録します。
- 不変条件（AGENTS.md §5）を緩める変更は、必ずユーザーの承認を得てから行います。
