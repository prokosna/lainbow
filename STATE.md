# STATE

> このファイルを編集するのは **Supervisor だけ**です。WI のレビューが終わったら、他の作業より先にここを更新してください。
> 到着した Supervisor は、このファイルを読んだあと `git log --oneline -15` を実行し、`last_commit` より新しいコミットがあれば、まずここを実態に合わせます。
> **150 行を超えないようにします。** 古い Review log は `docs/plan/review-log.md` に移します（短く書き直して消すことはしません）。

```yaml
last_updated: 2026-10-18
last_commit: 600efa1          # この計画を書いた時点の HEAD。計画のコミット後に更新すること
current_milestone: M0
current_wi: none
branch: main                  # M0 の開始時に improve/m0-agent-workflow を作る
```

## 次の 3 アクション

1. ブランチ `improve/m0-agent-workflow` を作り、計画ドキュメントをコミットする（ユーザーの指示を待つ）
2. WI-M0-01: `scripts/dispatch-implementer.sh` と `scripts/scout.sh` を作る（Supervisor が直接実装）
3. WI-M0-02: `scripts/check.sh` を作り、ruff / mypy のベースラインをここに記録する

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

## ゲートのベースライン

（WI-M0-02 で記録する）

## Design changes

- 2026-10-18: 計画書の初版を作成（docs/plan/improvement-plan.md）
- 2026-10-18: Q1〜Q4 の回答を反映。D-A を承認済みにし、D-F（既定モデル）を追加。M2-02 に pinned モデルを、M3-04 に `LAINBOW_ANALYZE_MODELS` を追加
- 2026-10-18: Q3 を変更。CPU と GPU でモデルの構成を揃え、既定はどちらも `muq`（類似曲検索）と `muq_mulan`（自然言語検索）とする。Q5 は解消

## Review log

| 日付 | WI | 判定 | モデル | 回数 | メモ |
|---|---|---|---|---|---|
