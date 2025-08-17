import logging
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)

from domain import config

logger = logging.getLogger(__name__)


def _connect_to_milvus() -> None:
    """Establishes a connection to Milvus if one does not already exist."""
    alias = config.MILVUS_CONNECTION_ALIAS
    if not connections.has_connection(alias):
        logger.info(f"Connecting to Milvus at {config.MILVUS_HOST}:{config.MILVUS_PORT}")
        try:
            connections.connect(alias=alias, host=config.MILVUS_HOST, port=config.MILVUS_PORT)
            logger.info("Successfully connected to Milvus.")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    else:
        logger.debug("Milvus connection already exists.")


def _get_client() -> MilvusClient:
    return MilvusClient(
        uri=f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}",
        token="root:Milvus",
    )


def disconnect_from_milvus() -> None:
    """Disconnects from Milvus if a connection exists."""
    alias = config.MILVUS_CONNECTION_ALIAS
    if connections.has_connection(alias):
        logger.info("Disconnecting from Milvus.")
        connections.disconnect(alias)
        logger.info("Successfully disconnected from Milvus.")
    else:
        logger.debug("No active Milvus connection to disconnect.")


def create_collection_if_not_exists(
    collection_name: str,
    dimension: int,
    id_field: str,
    vector_field: str,
    index_params: dict[str, Any],
) -> None:
    """
    Checks if the collection exists, and if not, creates it with the defined schema and index.
    Assumes that a connection to Milvus has already been established.
    """
    _connect_to_milvus()
    if not utility.has_collection(collection_name):
        logger.info(f"Collection '{collection_name}' not found. Creating it now.")
        # Define schema
        field_id = FieldSchema(
            name=id_field, dtype=DataType.VARCHAR, is_primary=True, max_length=512
        )
        field_vector = FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, dim=dimension)
        schema = CollectionSchema(
            fields=[field_id, field_vector],
            description=f"{collection_name} collection",
            enable_dynamic_field=False,
        )
        # Create collection
        collection = Collection(name=collection_name, schema=schema)
        logger.info(f"Collection '{collection_name}' created successfully.")
        # Create index
        logger.info("Creating index...")
        collection.create_index(field_name=vector_field, index_params=index_params)
        logger.info("Index created successfully.")
    else:
        logger.debug(f"Collection '{collection_name}' already exists.")


def upsert_vectors(collection_name: str, data: list[dict[str, Any]]) -> None:
    """
    Upserts (inserts or updates) vectors into the Milvus collection.

    Args:
        collection_name: The name of the collection to upsert into.
        ids: A list of integer song IDs.
        vectors: A list of numpy arrays representing the embeddings.
    """
    _connect_to_milvus()
    if not data:
        logger.info("No data provided for upsert.")
        return

    collection = Collection(collection_name)

    logger.info(f"Upserting {len(data)} entities into '{collection_name}'.")
    collection.upsert(data)
    collection.flush()  # Ensure data is written to disk
    logger.info(f"Successfully upserted {len(data)} entities.")


def delete_vectors(
    collection_name: str,
    ids: list[str],
    id_field: str,
) -> None:
    """
    Deletes vectors from the Milvus collection based on their IDs.

    Args:
        collection_name: The name of the collection to delete from.
        ids: A list of string IDs to delete.
        id_field: The name of the primary key field.
    """
    _connect_to_milvus()
    if not ids:
        logger.info("No vectors to delete.")
        return

    if not utility.has_collection(collection_name):
        logger.warning(f"Collection '{collection_name}' does not exist. Cannot delete vectors.")
        return

    collection = Collection(collection_name)
    # Convert list of ids to a string expression for the query
    expr = f"{id_field} in {ids}"
    logger.info(f"Deleting {len(ids)} vectors from '{collection_name}' with expression: {expr}")
    collection.delete(expr)
    collection.flush()
    logger.info(f"Successfully deleted {len(ids)} vectors.")


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
    _connect_to_milvus()
    collection = Collection(collection_name)
    collection.load()

    client = _get_client()
    resp = client.get(
        collection_name=collection_name,
        ids=[vector_id],
        output_fields=[vector_field],
    )
    if len(resp) == 0:
        logger.warning(f"Vector with ID '{vector_id}' not found in '{collection_name}'.")
        return None

    vector: list[float] = resp[0][vector_field]
    return vector


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
    _connect_to_milvus()
    collection = Collection(collection_name)
    collection.load()

    client = _get_client()
    resp = client.search(
        collection_name=collection_name,
        anns_field=vector_field,
        data=[query_vector],
        limit=limit,
    )
    if len(resp) == 0:
        logger.warning(f"No results found for query vector in '{collection_name}'.")
        return []

    results: list[dict[str, Any]] = resp[0]
    return results
