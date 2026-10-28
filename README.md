# Lainbow

Lainbow analyzes a local music library with audio models and serves similar-song search and
natural-language search. It is built for the MPD client [Sola MPD](https://github.com/prokosna/sola_mpd).

It runs as three containers: the server (API), the engine (analysis) and Postgres with pgvector.
The only state is the database.

## Quick start (one machine)

1. Clone the release (Lainbow publishes no images; every machine builds its own):

   ```sh
   git clone --branch v2.0.0 https://github.com/prokosna/lainbow.git && cd lainbow
   ./lainbow init
   docker compose up -d --build
   ```

   The first build takes several minutes.
   `./lainbow init` asks for the music folder, the data folder and the port, and can check the music folder against MPD.

2. Check that the server is up: `curl http://localhost:8000/health`.
3. In Sola MPD, set the Lainbow address to `http://<host>:8000`.

**Lainbow's music folder must be the same folder as MPD's `music_directory`.** Paths are relative to it.

## Day-to-day

```sh
docker compose up -d     # start
docker compose down      # stop
./lainbow update         # git fetch --tags, then rebuild this checkout and restart
./lainbow status         # containers, analysis progress and engines
```

To move to a newer release: `git checkout <tag> && ./lainbow update`.

New music is found by the hourly scan or by the scan button in Sola MPD, and is analyzed automatically.

## Speed up the first analysis with a GPU machine

The first full analysis is the slow part. Any machine with an NVIDIA or AMD GPU can help.

1. On the server, run `./lainbow add-engine desktop` (add `--host <address>` if the guessed address is wrong). It prints a `./lainbow join ...` line.
2. On the GPU machine, clone the same tag and run that line. The first `join` builds the engine image, which takes a while. `join` detects NVIDIA, AMD or CPU, checks that the server version matches, and starts the engine.
3. The engine only connects out to the server. Nothing needs to be opened or forwarded on that machine.
4. When the analysis is done, stop it with `./lainbow leave`. The engine stays registered and can rejoin later with the same line; `./lainbow remove-engine <name>` on the server removes it for good.

Steady-state analysis continues on the server.

An Apple Silicon Mac can join as a CPU engine (Docker Desktop); it cannot use the Mac's GPU from Docker.

Windows (WSL2):

1. Install the NVIDIA driver for Windows.
2. Run `wsl --install` and open an Ubuntu terminal.
3. Install Docker Desktop with the WSL2 backend and enable its WSL integration for Ubuntu.
4. Clone Lainbow inside WSL and run the `./lainbow join ...` line there.

## Hardware and time

Every song is analyzed once. Rough time for 100,000 songs:

- **GPU**: about 1.5 days (NVIDIA RTX 5070 Ti). Needs an NVIDIA GPU with 12 GB+ VRAM.
- **CPU only**: about 6 weeks (AMD Ryzen AI 9 HX 370). Needs about 10 GB of free RAM.

- Model weights (several GB) are downloaded on the engine's first start and kept in `DATA_DIR/models`.

## Configuration

Everything has a default except the first four keys of `.env`, which `./lainbow init` writes.
The full list is in [docs/configuration.md](docs/configuration.md). `./lainbow config` shows the effective values.
The settings most people touch:

- `MUSIC_DIR`: the music folder.
- `PORT`: the API port.
- `LAINBOW_MODELS`: which models to analyze (default `muq,muq_mulan`; `clap` and `mert` are optional).
- `LAINBOW_SCAN_INTERVAL_MIN`: minutes between automatic scans.

## API

- `GET /health`: liveness.
- `GET /api/v1/stats`: library and analysis counts.
- `GET /api/v1/songs/search`: natural-language search (`q`, `model_name`, `limit`).
- `GET /api/v1/songs/{path}/similar`: songs similar to a song (`model_name`, `limit`), ranked with CSLS
  (see [docs/design.md](docs/design.md#similar-song-ranking)).
- `GET /api/v1/songs/{path}/analysis`: analysis state of a song per model.
- `POST /api/v1/batch/{scan,vacuum,analyze}/run`: start a scan, a vacuum (also a scan) or a retry of failed tracks.
- `GET /api/v1/batch/tasks/{id}`: status of a run started above.

FastAPI's `/docs` has the full schema.

## Upgrading from v1.x

v2 is a rewrite and is not compatible with v1.x. Nothing is carried over: install it as new and let it analyze
the library again.

1. Stop v1 (`docker compose down` in its directory). Its data is not used and can be deleted once v2 works.
2. Install v2 with a new data directory, following [Quick start](#quick-start-one-machine).
3. Use Sola MPD 4.13.0 or later.

v1 stays available at [v1.0.0](https://github.com/prokosna/lainbow/releases/tag/v1.0.0).

## Development

```sh
uv sync --extra server --extra engine --extra models --extra optional-models --extra cpu
scripts/check.sh        # ruff, mypy, import-linter, pytest, shellcheck
scripts/gen-docs.sh     # regenerate docs/configuration.md
scripts/e2e-smoke.sh    # build the images and run compose end to end (needs docker)
```

The postgres tests run only when a disposable database is given:

```sh
docker run -d --rm --name lainbow_test_pg -e POSTGRES_USER=test -e POSTGRES_PASSWORD=test \
  -e POSTGRES_DB=lainbow_test -p 127.0.0.1:55432:5432 pgvector/pgvector:pg17
LAINBOW_TEST_DATABASE_URL=postgresql+psycopg://test:test@127.0.0.1:55432/lainbow_test uv run pytest -m postgres
```

Packages in `src/lainbow`:

- `core`: models, analysis methods and settings. It imports no other Lainbow package.
- `server`: the HTTP API, library scan and engine coordination.
- `db`: schema, migrations, vectors and work queue. It never imports `server`.
- `engine`: audio decoding and models. It talks to the server over HTTP only and never imports `server` or `db`.

See [docs/design.md](docs/design.md) for the design.

## License

[MIT](LICENSE)
