# Design

Lainbow analyzes a music library with audio models, stores one vector per track and model, and answers
nearest-neighbour queries. This page describes how it is put together and why.

## Services

```
Sola MPD --HTTP--> server --SQL--> postgres (pgvector)
                     ^
                     +------HTTP + token------ engine (CPU, CUDA or ROCm; any number)
```

- **Three services: server, engine, Postgres with pgvector.** Postgres is the only state, so a backup of one
  folder is a backup of Lainbow, and the server and engines can be replaced at will.
- **Images are built locally from the release checkout; nothing is published to a registry.**
- **The server** serves the API, scans the library (at start, hourly, or on request) and coordinates engines.
- **The engine** decodes audio, runs the models and posts vectors back. It also embeds search text for the server.
- **One image per accelerator** (`cpu`, `cuda`, `rocm`). `compose.yaml` runs exactly one, chosen by
  `COMPOSE_PROFILES`.

## Engines and work

- **Engines talk to the server only over HTTP, with a token.** An engine needs no database access and no
  inbound port, so a GPU machine on another network only has to reach the server. Every engine request carries
  `X-Lainbow-Version`, and a mismatch is a 409. The import rules enforce it: `engine` never imports `server` or
  `db`.
- **The work is derived from the database, not queued.** A track needs work when it has no vector for the active
  model and profile. Nothing can be lost or duplicated by a crashed queue, and adding a model or changing a
  method creates work by itself.
- **Work is leased per track.** An engine claims a few tracks for `LAINBOW_LEASE_SEC`. If it dies, the lease
  expires and another engine takes them. Several engines can work at once without coordination.
- **A track that keeps failing is marked failed** after a few attempts and is retried only on request
  (`POST /api/v1/batch/analyze/run`), so a broken file does not block the rest.

## Audio access

- **Audio is served over HTTP.** An engine downloads a track from the server (`/engine/v1/tracks/{id}/audio`), so a
  remote engine needs no mount of the library.
- **A local read when the engine can see the folder.** If `LAINBOW_MUSIC_DIR` is set on the engine and the file
  exists there, it is read directly. The bundled engine mounts the library read-only and avoids the transfer.

## Analysis method and profiles

- **The server owns the analysis method.** `LAINBOW_<MODEL>_MAX_SEC` and `LAINBOW_<MODEL>_DROP_TAIL` are read
  by the server and sent to engines in the plan, so all engines analyze the same way.
- **Every vector is labelled with a profile**, built from the model revision and the method
  (`r1-max180-keeptail`). Vectors of different profiles never mix in a query, and a changed method means new
  work rather than silently wrong results. Vectors of the previous profile are kept, so switching back is free.
- **The model revision** (in `ModelSpec`) is bumped whenever a code or weight change alters the vector, which
  invalidates the old ones the same way.
- **Weights are pinned** to a Hugging Face commit per model, so every engine downloads the same weights;
  changing a pin is a weight change and bumps the model revision.

## Similar-song ranking

- **Similar songs are ranked with CSLS**, not raw cosine similarity. Embeddings have hubs: tracks that are close
  to almost everything and turn up in most result lists. The server takes the 200 nearest candidates and ranks
  them by `2 * cos(q, c) - r(c)`, where `r(c)` is the mean similarity of `c` to its k nearest neighbours
  (`LAINBOW_SIMILAR_CSLS_K`, default 50). On a 94k-track library this cut the share of tracks that never appear
  in any top 10 from 4.1% to 0.5%, without hurting listening tests. The `score` in responses stays the cosine
  similarity.
- **The neighbour statistics are derived from the database, like the work.** A minute after start and every ten
  minutes the server checks each active model and profile, and computes `r` with the HNSW index when it is
  missing, when the library has grown by a quarter or shrunk by a fifth since the last full pass, or when k
  changed. New tracks in between get their own `r` on the next check. An upgrade therefore needs no step:
  until the statistics exist, search falls back to plain cosine similarity.
- **Literals for the profile in vector SQL.** HNSW indexes are partial, one per profile, so the profile and the
  limit are written into the SQL rather than bound; a generic plan of a prepared statement could otherwise not
  use the index.
- Natural-language search is unchanged.

## Data

- **Vector tables are created at runtime**, one per model and dimension, with an HNSW index per profile. The
  set of models and the vector sizes are code, not schema, so adding a model needs no migration.
- **Everything else is ordinary tables** (tracks, analysis state, runs, engines) managed by Alembic migrations
  that run at server start.
- **Paths are relative to the music folder**, the same as MPD's `music_directory`, so Sola MPD can address
  a track by the path it already knows.

## Extending

- **Adding a model touches `MODELS` in `core/models.py` and one engine module** in `engine/models`. The
  settings, work derivation, vector tables and API pick it up from the `MODELS` entry.
- **Adding a method parameter touches `core/method.py`.** A field on `AnalysisMethod` with a label token gets
  its environment variable, its place in the profile label and its row in `docs/configuration.md` automatically.
- **`docs/configuration.md` is generated** from the settings classes by `scripts/gen-docs.sh`, and a test fails
  when it is out of date.

## Package layout

| Package | Role | May import |
| --- | --- | --- |
| `core` | models, methods, settings | nothing else in Lainbow |
| `db` | schema, migrations, vectors, work queue | `core` |
| `server` | HTTP API, scan, engine coordination | `core`, `db` |
| `engine` | decoding, models, worker | `core` |

The rules are checked by import-linter as part of `scripts/check.sh`.

## Compatibility

The API used by Sola MPD (`/health`, `/api/v1/stats`, the song endpoints and the batch endpoints) keeps its
paths and response fields. It does not read data from v1.x.
