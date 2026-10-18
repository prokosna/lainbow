# Work Order <WI-ID> — <タイトル>（REVISE round N の場合はそう書く）

## Role

あなたは **Implementer** です。リポジトリの `AGENTS.md` §4 と §5 に従ってください。
`docs/`、`AGENTS.md`、`CLAUDE.md`、`STATE.md`、`.agent/` は**編集しないでください**。
**commit、push、checkout、reset、stash、ブランチ操作はしないでください。** 変更は作業ツリーに残してください。

## Context

- 計画書: `docs/plan/improvement-plan.md` の <節名>
- 背景（2〜5 行）: なぜこの変更が必要か。関係する不変条件（I1〜I5）があれば挙げる。

## Definition of Done

1. <ファイル>: <期待する挙動を具体的に。関数名、引数、戻り値、エラー時の扱い>
2. ...
3. テスト: <追加するテストのファイル名と、確認するケース>

## Out of scope（やらないこと）

- <隣接していても触らないファイルや関数>
- DoD に書かれていないリファクタリング、依存関係の追加

## Gates

```
scripts/check.sh
<WI 固有のゲート>
```

## Report back

- 変更したファイルの一覧
- 各ゲートの出力の末尾（そのまま貼る）。実行していないものは `NOT RUN`
- Design Change Request があれば（なければ "none"）
