from collections import Counter
from typing import Any

from api.core.text_embedding import get_text_embedding_service
from api.session_manager_wrap import get_db_session
from domain import milvus_utils
from domain.schemas import EmbeddingModel, Song, SongEmbedding, SongFeatures, TextEmbeddingModel
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlmodel import Session, col, select
from worker import acoustic_features, clap, mert, muq, muq_mulan

MODEL_CONFIGS = {
    EmbeddingModel.ACOUSTIC_FEATURES: {
        "collection_name": acoustic_features.COLLECTION_NAME,
        "id_field": acoustic_features.ID_FIELD,
        "vector_field": acoustic_features.VECTOR_FIELD,
    },
    EmbeddingModel.CLAP: {
        "collection_name": clap.COLLECTION_NAME,
        "id_field": clap.ID_FIELD,
        "vector_field": clap.VECTOR_FIELD,
    },
    EmbeddingModel.MERT: {
        "collection_name": mert.COLLECTION_NAME,
        "id_field": mert.ID_FIELD,
        "vector_field": mert.VECTOR_FIELD,
    },
    EmbeddingModel.MUQ: {
        "collection_name": muq.COLLECTION_NAME,
        "id_field": muq.ID_FIELD,
        "vector_field": muq.VECTOR_FIELD,
    },
    EmbeddingModel.MUQ_MULAN: {
        "collection_name": muq_mulan.COLLECTION_NAME,
        "id_field": muq_mulan.ID_FIELD,
        "vector_field": muq_mulan.VECTOR_FIELD,
    },
}

router = APIRouter()


class SongAnalysis(BaseModel):
    """Response model for detailed song analysis."""

    # From Song
    file_path: str
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    duration_seconds: int | None = None

    # Status
    has_features: bool
    embeddings_status: dict[str, str | None]

    # From SongFeatures
    bpm: float | None = None
    spectral_centroid_mean: float | None = None
    spectral_bandwidth_mean: float | None = None
    mfcc_mean: list[float] | None = None
    chroma_mean: list[float] | None = None


@router.get(
    "/{file_path:path}/analysis",
    response_model=SongAnalysis,
    summary="Get Song Analysis",
    description="Retrieves detailed analysis for a given song (BPM, key, chords, etc.).",
)
def get_song_analysis(
    file_path: str,
    db: Session = Depends(get_db_session),  # noqa: B008
) -> SongAnalysis:
    """
    Retrieves detailed analysis for a given song.
    """
    song = db.exec(select(Song).where(col(Song.file_path) == file_path)).one_or_none()

    if not song:
        raise HTTPException(status_code=404, detail="Song not found")

    features = db.exec(
        select(SongFeatures).where(col(SongFeatures.file_path) == file_path)
    ).one_or_none()
    embeddings = db.exec(
        select(SongEmbedding).where(col(SongEmbedding.file_path) == file_path)
    ).all()

    embedding_status_dict: dict[str, str | None] = {model.value: None for model in EmbeddingModel}
    for emb in embeddings:
        embedding_status_dict[emb.model_name.value] = emb.status.value

    # Prepare feature lists
    mfcc_means = []
    chroma_means = []
    if features:
        mfcc_means = [getattr(features, f"mfcc_mean_{i}", 0.0) for i in range(13)]
        chroma_means = [getattr(features, f"chroma_mean_{i}", 0.0) for i in range(12)]

    return SongAnalysis(
        file_path=song.file_path,
        title=song.title,
        artist=song.artist,
        album=song.album,
        duration_seconds=song.duration_seconds,
        has_features=features is not None,
        embeddings_status=embedding_status_dict,
        bpm=features.bpm if features else None,
        spectral_centroid_mean=features.spectral_centroid_mean if features else None,
        spectral_bandwidth_mean=features.spectral_bandwidth_mean if features else None,
        mfcc_mean=mfcc_means if features else None,
        chroma_mean=chroma_means if features else None,
    )


class SearchSong(BaseModel):
    """Response model for a single similar song."""

    file_path: str
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    genre: str | None = None
    score: float


class SearchSongResult(BaseModel):
    """Response model for a list of similar songs."""

    songs: list[SearchSong]


@router.get(
    "/{file_path:path}/similar",
    response_model=SearchSongResult,
    summary="Find Similar Songs",
    description="Finds songs similar to the given song based on a selected embedding model.",
)
def find_similar_songs(
    file_path: str,
    model_name: EmbeddingModel = Query(  # noqa: B008
        default=EmbeddingModel.MUQ,
        description="The embedding model to use for similarity search.",
    ),
    limit: int = Query(default=10),
    db: Session = Depends(get_db_session),  # noqa: B008
) -> SearchSongResult:
    """
    Finds songs acoustically similar to the given song using a specified vector embedding.
    """
    config = MODEL_CONFIGS[model_name]
    collection_name = config["collection_name"]
    id_field = config["id_field"]
    vector_field = config["vector_field"]

    query_vec = milvus_utils.query_vector(
        collection_name=collection_name,
        vector_id=file_path,
        vector_field=vector_field,
    )
    if not query_vec:
        raise HTTPException(
            status_code=404,
            detail=f"Vector for model '{model_name.value}' not found for this song.",
        )

    search_results = milvus_utils.search_vectors(
        collection_name=collection_name,
        query_vector=query_vec,
        vector_field=vector_field,
        limit=limit + 1,
    )

    similar_songs_data: list[dict[str, Any]] = []
    for hit in search_results:
        if hit[id_field] != file_path:
            similar_songs_data.append({"file_path": hit[id_field], "score": hit["distance"]})

    similar_songs_data = similar_songs_data[:limit]

    if not similar_songs_data:
        return SearchSongResult(songs=[])

    similar_file_paths = [data["file_path"] for data in similar_songs_data]
    songs_from_db = db.exec(select(Song).where(col(Song.file_path).in_(similar_file_paths))).all()
    songs_map = {song.file_path: song for song in songs_from_db}

    response_songs = []
    for data in similar_songs_data:
        song_info = songs_map.get(data["file_path"])
        if song_info:
            response_songs.append(
                SearchSong(
                    file_path=song_info.file_path,
                    title=song_info.title,
                    artist=song_info.artist,
                    album=song_info.album,
                    genre=song_info.genre,
                    score=data["score"],
                )
            )

    return SearchSongResult(songs=response_songs)


@router.get(
    "/search",
    response_model=SearchSongResult,
    summary="Search Songs by Natural Language",
    description="Searches for songs based on a natural language query, e.g., 'a song for a summer evening'.",
)
def search_songs_by_natural_language(
    q: str = Query(..., description="Natural language query for song search."),  # noqa: B008
    model_name: TextEmbeddingModel = Query(  # noqa: B008
        default=TextEmbeddingModel.MUQ_MULAN,
        description="The text embedding model to use for the search.",
    ),
    limit: int = Query(default=10, description="Maximum number of results to return."),
    db: Session = Depends(get_db_session),  # noqa: B008
) -> SearchSongResult:
    """
    Searches for songs using a natural language query by embedding the query text
    and finding the most similar songs in the selected embedding space.
    """
    text_embedding_service = get_text_embedding_service()
    text_embed = text_embedding_service.get_text_embedding(q, model_name)

    if model_name == TextEmbeddingModel.CLAP:
        config = MODEL_CONFIGS[EmbeddingModel.CLAP]
    elif model_name == TextEmbeddingModel.MUQ_MULAN:
        config = MODEL_CONFIGS[EmbeddingModel.MUQ_MULAN]
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model_name}")

    collection_name = config["collection_name"]
    id_field = config["id_field"]
    vector_field = config["vector_field"]

    search_results = milvus_utils.search_vectors(
        collection_name=collection_name,
        query_vector=text_embed.tolist(),
        vector_field=vector_field,
        limit=limit,
    )

    if not search_results:
        return SearchSongResult(songs=[])

    search_hits_data = [
        {"file_path": hit[id_field], "score": hit["distance"]} for hit in search_results
    ]

    hit_file_paths = [hit["file_path"] for hit in search_hits_data]

    # Check for duplicates
    if len(hit_file_paths) != len(set(hit_file_paths)):
        print("Duplicate file paths found in search results:")
        counts = Counter(hit_file_paths)
        for path, count in counts.items():
            if count > 1:
                print(f"- {path} (found {count} times)")

    songs_from_db = db.exec(select(Song).where(col(Song.file_path).in_(hit_file_paths))).all()
    songs_map = {song.file_path: song for song in songs_from_db}

    response_songs = []
    for data in search_hits_data:
        song_info = songs_map.get(data["file_path"])
        if song_info:
            response_songs.append(
                SearchSong(
                    file_path=song_info.file_path,
                    title=song_info.title,
                    artist=song_info.artist,
                    album=song_info.album,
                    genre=song_info.genre,
                    score=data["score"],
                )
            )

    return SearchSongResult(songs=response_songs)
