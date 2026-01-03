import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from domain import config, vector_store_utils
from domain.schemas import SongFeatures, Status
from domain.session_manager import db_session_context
from sqlmodel import col, func, select

from worker import audio

logger = logging.getLogger(__name__)

COLLECTION_NAME = "acoustic_features"
EMBEDDING_DIMENSION = 55
ID_FIELD = "file_path"
VECTOR_FIELD = "feature_vector"
INDEX_PARAMS = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 256},
}
FEATURE_ORDER = (
    [
        "bpm",
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_std",
    ]
    + [f"mfcc_mean_{i}" for i in range(13)]
    + [f"mfcc_std_{i}" for i in range(13)]
    + [f"chroma_mean_{i}" for i in range(12)]
    + [f"chroma_std_{i}" for i in range(12)]
)
SAMPLING_RATE = 44100


@dataclass
class FeatureStat:
    """Dataclass to hold the mean and standard deviation of a single feature."""

    mean: float = 0.0
    std: float = 1.0


@dataclass
class SongFeaturesStats:
    """Dataclass to hold statistics for all song features."""

    bpm: FeatureStat = field(default_factory=FeatureStat)
    spectral_centroid_mean: FeatureStat = field(default_factory=FeatureStat)
    spectral_centroid_std: FeatureStat = field(default_factory=FeatureStat)
    spectral_bandwidth_mean: FeatureStat = field(default_factory=FeatureStat)
    spectral_bandwidth_std: FeatureStat = field(default_factory=FeatureStat)

    # MFCC stats
    mfcc_mean_0: FeatureStat = field(default_factory=FeatureStat)
    mfcc_std_0: FeatureStat = field(default_factory=FeatureStat)
    mfcc_mean_1: FeatureStat = field(default_factory=FeatureStat)
    mfcc_std_1: FeatureStat = field(default_factory=FeatureStat)
    mfcc_mean_2: FeatureStat = field(default_factory=FeatureStat)
    mfcc_std_2: FeatureStat = field(default_factory=FeatureStat)
    mfcc_mean_3: FeatureStat = field(default_factory=FeatureStat)
    mfcc_std_3: FeatureStat = field(default_factory=FeatureStat)
    mfcc_mean_4: FeatureStat = field(default_factory=FeatureStat)
    mfcc_std_4: FeatureStat = field(default_factory=FeatureStat)
    mfcc_mean_5: FeatureStat = field(default_factory=FeatureStat)
    mfcc_std_5: FeatureStat = field(default_factory=FeatureStat)
    mfcc_mean_6: FeatureStat = field(default_factory=FeatureStat)
    mfcc_std_6: FeatureStat = field(default_factory=FeatureStat)
    mfcc_mean_7: FeatureStat = field(default_factory=FeatureStat)
    mfcc_std_7: FeatureStat = field(default_factory=FeatureStat)
    mfcc_mean_8: FeatureStat = field(default_factory=FeatureStat)
    mfcc_std_8: FeatureStat = field(default_factory=FeatureStat)
    mfcc_mean_9: FeatureStat = field(default_factory=FeatureStat)
    mfcc_std_9: FeatureStat = field(default_factory=FeatureStat)
    mfcc_mean_10: FeatureStat = field(default_factory=FeatureStat)
    mfcc_std_10: FeatureStat = field(default_factory=FeatureStat)
    mfcc_mean_11: FeatureStat = field(default_factory=FeatureStat)
    mfcc_std_11: FeatureStat = field(default_factory=FeatureStat)
    mfcc_mean_12: FeatureStat = field(default_factory=FeatureStat)
    mfcc_std_12: FeatureStat = field(default_factory=FeatureStat)

    # Chroma stats
    chroma_mean_0: FeatureStat = field(default_factory=FeatureStat)
    chroma_std_0: FeatureStat = field(default_factory=FeatureStat)
    chroma_mean_1: FeatureStat = field(default_factory=FeatureStat)
    chroma_std_1: FeatureStat = field(default_factory=FeatureStat)
    chroma_mean_2: FeatureStat = field(default_factory=FeatureStat)
    chroma_std_2: FeatureStat = field(default_factory=FeatureStat)
    chroma_mean_3: FeatureStat = field(default_factory=FeatureStat)
    chroma_std_3: FeatureStat = field(default_factory=FeatureStat)
    chroma_mean_4: FeatureStat = field(default_factory=FeatureStat)
    chroma_std_4: FeatureStat = field(default_factory=FeatureStat)
    chroma_mean_5: FeatureStat = field(default_factory=FeatureStat)
    chroma_std_5: FeatureStat = field(default_factory=FeatureStat)
    chroma_mean_6: FeatureStat = field(default_factory=FeatureStat)
    chroma_std_6: FeatureStat = field(default_factory=FeatureStat)
    chroma_mean_7: FeatureStat = field(default_factory=FeatureStat)
    chroma_std_7: FeatureStat = field(default_factory=FeatureStat)
    chroma_mean_8: FeatureStat = field(default_factory=FeatureStat)
    chroma_std_8: FeatureStat = field(default_factory=FeatureStat)
    chroma_mean_9: FeatureStat = field(default_factory=FeatureStat)
    chroma_std_9: FeatureStat = field(default_factory=FeatureStat)
    chroma_mean_10: FeatureStat = field(default_factory=FeatureStat)
    chroma_std_10: FeatureStat = field(default_factory=FeatureStat)
    chroma_mean_11: FeatureStat = field(default_factory=FeatureStat)
    chroma_std_11: FeatureStat = field(default_factory=FeatureStat)


def calculate_features_stats() -> SongFeaturesStats:
    """Calculate and return the mean and std for each feature in a single query."""
    # List all feature columns explicitly for the query
    feature_columns = (
        [
            SongFeatures.bpm,
            SongFeatures.spectral_centroid_mean,
            SongFeatures.spectral_centroid_std,
            SongFeatures.spectral_bandwidth_mean,
            SongFeatures.spectral_bandwidth_std,
        ]
        + [getattr(SongFeatures, f"mfcc_mean_{i}") for i in range(13)]
        + [getattr(SongFeatures, f"mfcc_std_{i}") for i in range(13)]
        + [getattr(SongFeatures, f"chroma_mean_{i}") for i in range(12)]
        + [getattr(SongFeatures, f"chroma_std_{i}") for i in range(12)]
    )

    aggregations = [func.avg(c) for c in feature_columns] + [
        func.stddev(c) for c in feature_columns
    ]
    query = select(*aggregations).where(col(SongFeatures.status) == Status.COMPLETED)

    with db_session_context() as session:
        results = session.exec(query).one_or_none()
        if results is None:
            raise ValueError("Failed to calculate feature stats: no data found.")

        num_features = len(feature_columns)
        means = results[:num_features]
        stds = results[num_features:]

        stats_data = {}
        for i, column in enumerate(feature_columns):
            mean = means[i]
            std = stds[i]
            if mean is None or std is None:
                raise ValueError(f"Could not compute stats for feature '{column.key}'.")
            stats_data[column.key] = FeatureStat(mean=float(mean), std=float(std))

        return SongFeaturesStats(**stats_data)


def extract_acoustic_features(file_path: str) -> SongFeatures | None:
    """Extracts various acoustic features from a song file using librosa."""
    try:
        full_path = Path(config.MUSIC_NAS_ROOT_DIR) / file_path
        audio_file = audio.load_audio_for_librosa(full_path, sr=SAMPLING_RATE)
        y, sr = librosa.load(audio_file, sr=SAMPLING_RATE, mono=True)

        # Tempo and beats
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # Chroma features
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

        # Aggregate features
        mfcc_means = np.mean(mfcc, axis=1)
        mfcc_stds = np.std(mfcc, axis=1)
        chroma_means = np.mean(chroma_stft, axis=1)
        chroma_stds = np.std(chroma_stft, axis=1)

        features = SongFeatures(
            file_path=str(full_path.relative_to(config.MUSIC_NAS_ROOT_DIR)),
            status=Status.COMPLETED,
            bpm=float(tempo),
            spectral_centroid_mean=float(np.mean(spectral_centroid)),
            spectral_centroid_std=float(np.std(spectral_centroid)),
            spectral_bandwidth_mean=float(np.mean(spectral_bandwidth)),
            spectral_bandwidth_std=float(np.std(spectral_bandwidth)),
            mfcc_mean_0=float(mfcc_means[0]),
            mfcc_std_0=float(mfcc_stds[0]),
            mfcc_mean_1=float(mfcc_means[1]),
            mfcc_std_1=float(mfcc_stds[1]),
            mfcc_mean_2=float(mfcc_means[2]),
            mfcc_std_2=float(mfcc_stds[2]),
            mfcc_mean_3=float(mfcc_means[3]),
            mfcc_std_3=float(mfcc_stds[3]),
            mfcc_mean_4=float(mfcc_means[4]),
            mfcc_std_4=float(mfcc_stds[4]),
            mfcc_mean_5=float(mfcc_means[5]),
            mfcc_std_5=float(mfcc_stds[5]),
            mfcc_mean_6=float(mfcc_means[6]),
            mfcc_std_6=float(mfcc_stds[6]),
            mfcc_mean_7=float(mfcc_means[7]),
            mfcc_std_7=float(mfcc_stds[7]),
            mfcc_mean_8=float(mfcc_means[8]),
            mfcc_std_8=float(mfcc_stds[8]),
            mfcc_mean_9=float(mfcc_means[9]),
            mfcc_std_9=float(mfcc_stds[9]),
            mfcc_mean_10=float(mfcc_means[10]),
            mfcc_std_10=float(mfcc_stds[10]),
            mfcc_mean_11=float(mfcc_means[11]),
            mfcc_std_11=float(mfcc_stds[11]),
            mfcc_mean_12=float(mfcc_means[12]),
            mfcc_std_12=float(mfcc_stds[12]),
            chroma_mean_0=float(chroma_means[0]),
            chroma_std_0=float(chroma_stds[0]),
            chroma_mean_1=float(chroma_means[1]),
            chroma_std_1=float(chroma_stds[1]),
            chroma_mean_2=float(chroma_means[2]),
            chroma_std_2=float(chroma_stds[2]),
            chroma_mean_3=float(chroma_means[3]),
            chroma_std_3=float(chroma_stds[3]),
            chroma_mean_4=float(chroma_means[4]),
            chroma_std_4=float(chroma_stds[4]),
            chroma_mean_5=float(chroma_means[5]),
            chroma_std_5=float(chroma_stds[5]),
            chroma_mean_6=float(chroma_means[6]),
            chroma_std_6=float(chroma_stds[6]),
            chroma_mean_7=float(chroma_means[7]),
            chroma_std_7=float(chroma_stds[7]),
            chroma_mean_8=float(chroma_means[8]),
            chroma_std_8=float(chroma_stds[8]),
            chroma_mean_9=float(chroma_means[9]),
            chroma_std_9=float(chroma_stds[9]),
            chroma_mean_10=float(chroma_means[10]),
            chroma_std_10=float(chroma_stds[10]),
            chroma_mean_11=float(chroma_means[11]),
            chroma_std_11=float(chroma_stds[11]),
        )
        return features

    except Exception as e:
        logger.error(f"Failed to extract features for {file_path}: {e}")
        return None


def create_collection_if_not_exists() -> None:
    """
    Creates the vector collection if it doesn't exist.
    """
    vector_store_utils.create_collection_if_not_exists(
        collection_name=COLLECTION_NAME,
        dimension=EMBEDDING_DIMENSION,
        id_field=ID_FIELD,
        vector_field=VECTOR_FIELD,
        index_params=INDEX_PARAMS,
    )
    logger.info(f"Vector collection '{COLLECTION_NAME}' is ready.")


def delete_vectors(file_paths: list[str]) -> None:
    """
    Deletes vectors from the song features vector collection based on their file paths.
    """
    if not file_paths:
        logger.debug("No file paths provided to delete from vector store.")
        return
    try:
        logger.info(
            f"Attempting to delete {len(file_paths)} vectors from vector collection '{COLLECTION_NAME}'"
        )
        vector_store_utils.delete_vectors(
            collection_name=COLLECTION_NAME,
            ids=file_paths,
            id_field=ID_FIELD,
        )
        logger.info(f"Successfully deleted vectors for {len(file_paths)} file paths.")
    except Exception as e:
        raise Exception(f"Failed to delete vectors from vector store: {e}") from e


def create_feature_vector_data(
    song_features: SongFeatures,
    stats: SongFeaturesStats,
) -> dict[str, Any]:
    """
    Creates a standardized and normalized vector from SongFeatures and returns
    a dictionary formatted for vector store upsert.

    Args:
        song_features: The raw features for a single song.
        stats: The statistics (mean, std) for all features across the dataset.

    Returns:
        A dictionary containing the file_path and the normalized feature_vector.
    """
    raw_vector = np.array(
        [getattr(song_features, name) for name in FEATURE_ORDER], dtype=np.float32
    )
    mean_vector = np.array([getattr(stats, name).mean for name in FEATURE_ORDER], dtype=np.float32)
    std_vector = np.array([getattr(stats, name).std for name in FEATURE_ORDER], dtype=np.float32)

    std_vector[std_vector == 0] = 1.0

    standardized_vector = (raw_vector - mean_vector) / std_vector

    norm = np.linalg.norm(standardized_vector)
    normalized_vector = standardized_vector if norm == 0 else standardized_vector / norm

    return {
        ID_FIELD: song_features.file_path,
        VECTOR_FIELD: normalized_vector.tolist(),
    }
