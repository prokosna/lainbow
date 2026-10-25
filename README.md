# Lainbow

Lainbow analyzes a local music library and serves similar-song and natural-language search,
primarily for the MPD client [Sola MPD](https://github.com/prokosna/sola_mpd).

> **Lainbow 2 is under development on this branch.** It is a rewrite and is not compatible with
> Lainbow 1. The last release of Lainbow 1 is [v1.0.0](https://github.com/prokosna/lainbow/releases/tag/v1.0.0).

## Development

```sh
uv sync --all-extras
scripts/check.sh   # ruff, mypy (strict), import-linter, pytest, shellcheck
```

## License

[MIT](LICENSE)
