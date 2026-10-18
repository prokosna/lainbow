# STATE

> このファイルを編集するのは **Supervisor だけ**です。WI のレビューが終わったら、他の作業より先にここを更新してください。
> 到着した Supervisor は、このファイルを読んだあと `git log --oneline -15` を実行し、`last_commit` より新しいコミットがあれば、まずここを実態に合わせます。
> **150 行を超えないようにします。** 古い Review log は `docs/plan/review-log.md` に移します（短く書き直して消すことはしません）。

```yaml
last_updated: 2026-10-18
last_commit: 2d06bfc          # docs: add improvement plan（M0-01 のコミットはこの次）
current_milestone: M0
current_wi: WI-M0-02
branch: improve/m0-agent-workflow
```

## 次の 3 アクション

1. WI-M0-02: `scripts/check.sh` を作り、ruff / mypy のベースラインをここに記録する（Supervisor が直接実装）
2. WI-M0-03: タスクロジックのテスト基盤（ここから Implementer へのディスパッチを使い始める）
3. M1 の Work Order を順に発行する（M1-01 は Supervisor がテストを先に書く）

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

（WI-M0-02 で記録する）

## Design changes

- 2026-10-18: 計画書の初版を作成（docs/plan/improvement-plan.md）
- 2026-10-18: Q1〜Q4 の回答を反映。D-A を承認済みにし、D-F（既定モデル）を追加。M2-02 に pinned モデルを、M3-04 に `LAINBOW_ANALYZE_MODELS` を追加
- 2026-10-18: Q3 を変更。CPU と GPU でモデルの構成を揃え、既定はどちらも `muq`（類似曲検索）と `muq_mulan`（自然言語検索）とする。Q5 は解消

## Review log

| 日付 | WI | 判定 | モデル | 回数 | メモ |
|---|---|---|---|---|---|
| 2026-10-18 | WI-M0-01 | PASS | Supervisor（直接実装） | - | dispatch / scout スクリプトを作成。shellcheck で問題なし、probe で動作確認 |
