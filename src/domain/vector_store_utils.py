import logging
from typing import Any
from uuid import UUID, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from domain import config

logger = logging.getLogger(__name__)


QDRANT_ID_NAMESPACE = UUID("4e7c4f4d-0a4c-4f4f-9a32-3d7a3f0a6c3d")


def _get_client() -> QdrantClient:
    return QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)


def _qdrant_point_id_from_file_path(file_path: str) -> UUID:
    return uuid5(QDRANT_ID_NAMESPACE, file_path)


def disconnect_from_vector_store() -> None:
    """No-op for compatibility (Qdrant client is stateless)."""
    return


def create_collection_if_not_exists(
    collection_name: str,
    dimension: int,
    id_field: str,
    vector_field: str,
    index_params: dict[str, Any],
) -> None:
    """
    Checks if the collection exists, and if not, creates it with the defined schema and index.
    Assumes that the vector store is available.
    """
    # Deprecated: Used to be used for Milvus.
    _ = id_field
    _ = vector_field
    _ = index_params

    client = _get_client()
    if client.collection_exists(collection_name=collection_name):
        logger.debug(f"Collection '{collection_name}' already exists.")
        return

    logger.info(f"Collection '{collection_name}' not found. Creating it now.")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
    )
    logger.info(f"Collection '{collection_name}' created successfully.")


def upsert_vectors(collection_name: str, data: list[dict[str, Any]]) -> None:
    """
    Upserts (inserts or updates) vectors into the vector collection.

    Args:
        collection_name: The name of the collection to upsert into.
        ids: A list of integer song IDs.
        vectors: A list of numpy arrays representing the embeddings.
    """
    if not data:
        logger.info("No data provided for upsert.")
        return

    client = _get_client()
    if not client.collection_exists(collection_name=collection_name):
        raise ValueError(f"Collection '{collection_name}' does not exist.")

    points: list[PointStruct] = []
    for row in data:
        file_path_any = row.get("file_path")
        if not isinstance(file_path_any, str) or not file_path_any:
            raise ValueError(
                f"Invalid file_path for upsert into '{collection_name}': {file_path_any!r}"
            )

        vector_any = row.get("embedding", row.get("feature_vector"))
        if vector_any is None:
            raise ValueError(
                f"Missing vector for upsert into '{collection_name}': file_path={file_path_any}"
            )

        if hasattr(vector_any, "tolist"):
            vector_any = vector_any.tolist()
        vector = [float(x) for x in vector_any]

        points.append(
            PointStruct(
                id=_qdrant_point_id_from_file_path(file_path_any),
                vector=vector,
                payload={"file_path": file_path_any},
            )
        )

    logger.info(f"Upserting {len(points)} entities into '{collection_name}'.")
    client.upsert(collection_name=collection_name, points=points, wait=True)
    logger.info(f"Successfully upserted {len(points)} entities.")


def delete_vectors(
    collection_name: str,
    ids: list[str],
    id_field: str,
) -> None:
    """
    Deletes vectors from the vector collection based on their IDs.

    Args:
        collection_name: The name of the collection to delete from.
        ids: A list of string IDs to delete.
        id_field: The name of the primary key field.
    """
    if not ids:
        logger.info("No vectors to delete.")
        return

    # Deprecated: Used to be used for Milvus.
    _ = id_field

    client = _get_client()
    if not client.collection_exists(collection_name=collection_name):
        logger.warning(f"Collection '{collection_name}' does not exist. Cannot delete vectors.")
        return

    qdrant_ids = [_qdrant_point_id_from_file_path(file_path) for file_path in ids]
    logger.info(f"Deleting {len(qdrant_ids)} vectors from '{collection_name}'.")
    client.delete(collection_name=collection_name, points_selector=qdrant_ids, wait=True)
    logger.info(f"Successfully deleted {len(qdrant_ids)} vectors.")


def query_vector(
    collection_name: str,
    vector_id: str,
    vector_field: str,
) -> list[float] | None:
    """
    Queries for a single vector by its ID.

    Args:
        collection_name: The name of the collection to query from.
        vector_id: The ID of the vector to retrieve.
        vector_field: The name of the vector field.

    Returns:
        The vector if found, otherwise None.
    """
    client = _get_client()
    if not client.collection_exists(collection_name=collection_name):
        return None

    point_id = _qdrant_point_id_from_file_path(vector_id)
    records = client.retrieve(
        collection_name=collection_name,
        ids=[point_id],
        with_payload=False,
        with_vectors=True,
    )
    if not records:
        logger.warning(f"Vector with ID '{vector_id}' not found in '{collection_name}'.")
        return None

    vec_any = records[0].vector
    if isinstance(vec_any, dict):
        vec_any = vec_any[vector_field] if vector_field in vec_any else next(iter(vec_any.values()))
    return [float(x) for x in vec_any]


def search_vectors(
    collection_name: str,
    query_vector: list[float],
    vector_field: str,
    limit: int,
) -> list[dict[str, Any]]:
    """
    Performs a similarity search on the collection.

    Args:
        collection_name: The name of the collection to search in.
        query_vector: The vector to search for.
        vector_field: The name of the vector field.
        limit: The number of results to return.

    Returns:
        A list of search results. ({"file_path": xx, "distance": yy})
    """
    # Deprecated: Used to be used for Milvus.
    _ = vector_field

    client = _get_client()
    if not client.collection_exists(collection_name=collection_name):
        return []

    query_vector_f = [float(x) for x in query_vector]

    resp = client.query_points(
        collection_name=collection_name,
        query=query_vector_f,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    hits: Any = getattr(resp, "points", [])
    if not hits:
        logger.warning(f"No results found for query vector in '{collection_name}'.")
        return []

    results: list[dict[str, Any]] = []
    for hit in hits:
        payload = hit.payload or {}
        file_path = payload.get("file_path")
        if isinstance(file_path, str):
            results.append({"file_path": file_path, "distance": float(hit.score)})
    return results
