import argparse
import os
import sys
import time
from collections.abc import Iterator
from typing import Any
from uuid import UUID, uuid5

from pymilvus import Collection, connections, utility
from pymilvus.exceptions import MilvusException
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

COLLECTION_SPECS: list[dict[str, Any]] = [
    {
        "name": "acoustic_features",
        "dim": 55,
        "id_field": "file_path",
        "vector_field": "feature_vector",
    },
    {
        "name": "clap_audio_embeddings",
        "dim": 512,
        "id_field": "file_path",
        "vector_field": "embedding",
    },
    {
        "name": "mert_audio_embeddings",
        "dim": 1024,
        "id_field": "file_path",
        "vector_field": "embedding",
    },
    {
        "name": "muq_audio_embeddings",
        "dim": 1024,
        "id_field": "file_path",
        "vector_field": "embedding",
    },
    {
        "name": "muq_mulan_audio_embeddings",
        "dim": 512,
        "id_field": "file_path",
        "vector_field": "embedding",
    },
]


QDRANT_ID_NAMESPACE = UUID("4e7c4f4d-0a4c-4f4f-9a32-3d7a3f0a6c3d")


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value != "" else default


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(_env("MIGRATION_BATCH_SIZE", "512")),
    )
    parser.add_argument(
        "--collections",
        type=str,
        default="",
        help="Comma-separated collection names. Empty means all.",
    )
    parser.add_argument(
        "--milvus-load-retries",
        type=int,
        default=int(_env("MIGRATION_MILVUS_LOAD_RETRIES", "20")),
        help="Number of retries for Milvus collection load/query operations.",
    )
    parser.add_argument(
        "--milvus-load-retry-delay-sec",
        type=float,
        default=float(_env("MIGRATION_MILVUS_LOAD_RETRY_DELAY_SEC", "3.0")),
        help="Initial retry delay (seconds) for Milvus load/query operations.",
    )
    return parser.parse_args()


def _get_milvus_connection() -> tuple[str, str]:
    host = _env("MILVUS_HOST", _env("DATABASE_SERVER_ENDPOINT", "localhost"))
    port = _env("MILVUS_PORT_API", "19530")
    return host, port


def _get_qdrant_connection() -> tuple[str, int]:
    host = _env("QDRANT_HOST", _env("DATABASE_SERVER_ENDPOINT", "localhost"))
    port = int(_env("QDRANT_PORT_HTTP", "6333"))
    return host, port


def _sleep_with_backoff(delay_sec: float, attempt: int) -> None:
    time.sleep(delay_sec * (1.5**attempt))


def _load_collection_with_retry(
    collection: Collection,
    *,
    collection_name: str,
    retries: int,
    initial_delay_sec: float,
) -> None:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            collection.load()
            return
        except MilvusException as e:
            last_exc = e
            if attempt >= retries:
                break
            print(
                f"[warn] Milvus load failed for '{collection_name}' (attempt {attempt + 1}/{retries + 1}): {e}",
                file=sys.stderr,
            )
            _sleep_with_backoff(initial_delay_sec, attempt)
    assert last_exc is not None
    raise last_exc


def _iter_milvus_rows(
    collection_name: str,
    id_field: str,
    vector_field: str,
    batch_size: int,
    milvus_load_retries: int,
    milvus_load_retry_delay_sec: float,
) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    collection = Collection(collection_name)
    _load_collection_with_retry(
        collection,
        collection_name=collection_name,
        retries=milvus_load_retries,
        initial_delay_sec=milvus_load_retry_delay_sec,
    )

    total = int(collection.num_entities)
    offset = 0
    expr = f'{id_field} != ""'

    while True:
        rows: list[dict[str, Any]] = []
        last_exc: Exception | None = None
        for attempt in range(milvus_load_retries + 1):
            try:
                rows = collection.query(
                    expr=expr,
                    output_fields=[id_field, vector_field],
                    limit=batch_size,
                    offset=offset,
                )
                last_exc = None
                break
            except MilvusException as e:
                last_exc = e
                if attempt >= milvus_load_retries:
                    break
                print(
                    f"[warn] Milvus query failed for '{collection_name}' (attempt {attempt + 1}/{milvus_load_retries + 1}): {e}",
                    file=sys.stderr,
                )
                _sleep_with_backoff(milvus_load_retry_delay_sec, attempt)
        if last_exc is not None:
            raise last_exc
        if not rows:
            break

        yield total, rows
        offset += len(rows)


def _normalize_vector(vec: Any) -> list[float]:
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    return [float(x) for x in vec]


def _qdrant_point_id_from_file_path(file_path: str) -> UUID:
    return uuid5(QDRANT_ID_NAMESPACE, file_path)


def _migrate_collection(
    *,
    milvus_host: str,
    milvus_port: str,
    qdrant_host: str,
    qdrant_port: int,
    collection_name: str,
    dim: int,
    id_field: str,
    vector_field: str,
    batch_size: int,
    milvus_load_retries: int,
    milvus_load_retry_delay_sec: float,
) -> None:
    connections.connect(alias="default", host=milvus_host, port=int(milvus_port))
    try:
        if not utility.has_collection(collection_name):
            print(f"[skip] Milvus collection not found: {collection_name}")
            return

        qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)

        print(f"[qdrant] recreate collection: {collection_name} (dim={dim})")
        if qdrant.collection_exists(collection_name=collection_name):
            qdrant.delete_collection(collection_name=collection_name)
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        processed = 0
        start = time.time()

        for total, rows in _iter_milvus_rows(
            collection_name=collection_name,
            id_field=id_field,
            vector_field=vector_field,
            batch_size=batch_size,
            milvus_load_retries=milvus_load_retries,
            milvus_load_retry_delay_sec=milvus_load_retry_delay_sec,
        ):
            points: list[PointStruct] = []
            for row in rows:
                point_id_any = row.get(id_field)
                if not isinstance(point_id_any, str) or not point_id_any:
                    raise ValueError(
                        f"Invalid point id in {collection_name}: {id_field}={point_id_any!r}"
                    )

                vec_any = row.get(vector_field)
                if vec_any is None:
                    raise ValueError(
                        f"Missing vector in {collection_name}: id={point_id_any}, field={vector_field}"
                    )

                vec = _normalize_vector(vec_any)
                if len(vec) != dim:
                    raise ValueError(
                        f"Vector dim mismatch in {collection_name}: id={point_id_any}, got={len(vec)}, expected={dim}"
                    )
                points.append(
                    PointStruct(
                        id=_qdrant_point_id_from_file_path(point_id_any),
                        vector=vec,
                        payload={"file_path": point_id_any},
                    )
                )

            qdrant.upsert(collection_name=collection_name, points=points, wait=True)

            processed += len(points)
            elapsed = time.time() - start
            rate = processed / elapsed if elapsed > 0 else 0.0
            print(f"[milvus->qdrant] {collection_name}: {processed}/{total} ({rate:.1f} points/s)")

        print(f"[done] {collection_name}: migrated {processed} points")
    finally:
        connections.disconnect(alias="default")


def main() -> None:
    args = _parse_args()

    requested = {name.strip() for name in args.collections.split(",") if name.strip()}
    specs = (
        [spec for spec in COLLECTION_SPECS if spec["name"] in requested]
        if requested
        else COLLECTION_SPECS
    )

    if requested and not specs:
        print(f"No matching collections requested: {sorted(requested)}", file=sys.stderr)
        sys.exit(2)

    milvus_host, milvus_port = _get_milvus_connection()
    qdrant_host, qdrant_port = _get_qdrant_connection()

    print(f"Milvus: {milvus_host}:{milvus_port}")
    print(f"Qdrant: {qdrant_host}:{qdrant_port}")

    for spec in specs:
        _migrate_collection(
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            collection_name=spec["name"],
            dim=spec["dim"],
            id_field=spec["id_field"],
            vector_field=spec["vector_field"],
            batch_size=args.batch_size,
            milvus_load_retries=args.milvus_load_retries,
            milvus_load_retry_delay_sec=args.milvus_load_retry_delay_sec,
        )


if __name__ == "__main__":
    main()
