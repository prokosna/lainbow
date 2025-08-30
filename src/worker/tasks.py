import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from uuid import UUID

from celery import chain, chord, group
from celery.canvas import Signature
from domain import config, milvus_utils, task_manager
from domain.messages import AnalyzeSongTaskPayload, ScanTaskPayload, VacuumTaskPayload
from domain.schemas import (
    EmbeddingModel,
    Song,
    SongEmbedding,
    SongFeatures,
    Status,
    TaskResult,
    TaskStatus,
)
from domain.session_manager import db_session_context
from sqlmodel import Session, col, delete, select, update

from worker import acoustic_features, audio, clap, mert, muq, muq_mulan
from worker.celery_app import celery_app

logger = logging.getLogger(__name__)

TASK_CHUNK_SIZE = config.TASK_CHUNK_SIZE
VECTORIZE_CHUNK_SIZE = 1000


def _get_log_message(name: str, task_id: str, message: str) -> str:
    return f"[{name} {task_id}] {message}"


def _skip_task_if_already_running(db: Session, name: str, task_id: UUID, start: float) -> bool:
    existing_task = task_manager.get_latest_task_by_name(db, name=name, excluding_id=task_id)
    if existing_task is not None and existing_task.status in [
        TaskStatus.RUNNING,
        TaskStatus.PENDING,
    ]:
        logger.info(_get_log_message(name, str(task_id), "Task already running."))
        task_manager.update_task_status(
            db,
            task_id=task_id,
            status=TaskStatus.SKIPPED,
            elapsed_time=int(time.time() - start),
        )
        return True
    return False


@celery_app.task(queue="cpu_queue", ignore_result=True)
def vacuum_library_task(task_id: str, payload: dict[str, Any]) -> None:
    start = time.time()
    logger.info(_get_log_message(VacuumTaskPayload.NAME, task_id, "Received task."))
    try:
        task_uuid = UUID(task_id)
        VacuumTaskPayload(**payload)
    except Exception as e:
        logging.error(
            _get_log_message(VacuumTaskPayload.NAME, task_id, f"Failed to parse payload: {e}")
        )
        return

    try:
        with db_session_context() as db:
            if _skip_task_if_already_running(db, VacuumTaskPayload.NAME, task_uuid, start):
                return

            all_songs = db.exec(select(Song)).all()
            total_songs = len(all_songs)
            logger.info(
                _get_log_message(
                    VacuumTaskPayload.NAME, task_id, f"Found {total_songs} songs to check."
                )
            )

            if total_songs == 0:
                task_manager.mark_task_as_success(
                    db,
                    task_id=task_uuid,
                    message="No songs found in the database to vacuum.",
                    elapsed_time=int(time.time() - start),
                )
                return

            task_manager.update_task_status(
                db,
                task_id=task_uuid,
                status=TaskStatus.RUNNING,
                elapsed_time=int(time.time() - start),
            )

            processed_songs = 0
            deleted_count = 0
            updated_count = 0
            song_infos = [
                (Path(config.MUSIC_NAS_ROOT_DIR) / s.file_path, s.mtime) for s in all_songs
            ]

            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_info = {
                    executor.submit(audio.check_audio_file_status, info[0], info[1]): info
                    for info in song_infos
                }

                for future in as_completed(future_to_info):
                    result = future.result()
                    processed_songs += 1

                    if result.status == "NOT_EXIST":
                        file_path = str(
                            Path(result.file_path).relative_to(config.MUSIC_NAS_ROOT_DIR)
                        )
                        logger.info(f"'{file_path}' does not exist. Deleting metadata.")

                        # Delete from PostgreSQL
                        db.exec(delete(Song).where(Song.file_path == file_path))  # type: ignore
                        deleted_count += 1

                        # Delete from Milvus
                        acoustic_features.delete_vectors(file_paths=[file_path])
                        clap.delete_vectors(file_paths=[file_path])
                        mert.delete_vectors(file_paths=[file_path])
                        muq.delete_vectors(file_paths=[file_path])
                        muq_mulan.delete_vectors(file_paths=[file_path])
                    elif result.status == "STALE":
                        file_path = str(
                            Path(result.file_path).relative_to(config.MUSIC_NAS_ROOT_DIR)
                        )

                        assert (
                            result.metadata is not None
                        ), "Metadata should not be None for STALE status"

                        logger.info(
                            f"'{file_path}' is stale. Updating metadata and deleting embeddings."
                        )

                        # Delete related analysis and embeddings from PostgreSQL to mark for re-calculation
                        db.exec(delete(SongFeatures).where(SongFeatures.file_path == file_path))  # type: ignore
                        db.exec(delete(SongEmbedding).where(SongEmbedding.file_path == file_path))  # type: ignore

                        # Delete from Milvus
                        acoustic_features.delete_vectors(file_paths=[file_path])
                        clap.delete_vectors(file_paths=[file_path])
                        mert.delete_vectors(file_paths=[file_path])
                        muq.delete_vectors(file_paths=[file_path])
                        muq_mulan.delete_vectors(file_paths=[file_path])

                        # Then, update the song's metadata
                        db.exec(  # type: ignore
                            update(Song)
                            .where(col(Song.file_path) == file_path)
                            .values(
                                title=result.metadata.title,
                                artist=result.metadata.artist,
                                album=result.metadata.album,
                                genre=result.metadata.genre,
                                duration_seconds=result.metadata.duration_seconds,
                                mtime=result.metadata.mtime,
                            )
                        )
                        updated_count += 1

                    # Update progress periodically
                    if processed_songs % 50 == 0 or processed_songs == total_songs:
                        progress = int((processed_songs / total_songs) * 100)
                        task_manager.update_task_progress(
                            db,
                            task_id=task_uuid,
                            progress=progress,
                            elapsed_time=int(time.time() - start),
                        )
                        db.commit()

            logger.info(
                _get_log_message(
                    VacuumTaskPayload.NAME,
                    task_id,
                    f"Deleted {deleted_count} songs, Updated {updated_count} songs.",
                )
            )

            task_manager.mark_task_as_success(
                db,
                task_id=task_uuid,
                message=f"Vacuum complete. Deleted: {deleted_count}, Updated: {updated_count}.",
                elapsed_time=int(time.time() - start),
            )
    except Exception as e:
        logger.error(
            _get_log_message(VacuumTaskPayload.NAME, task_id, f"An unexpected error occurred: {e}"),
            exc_info=True,
        )
        try:
            with db_session_context() as db:
                task_manager.mark_task_as_failure(
                    db,
                    task_id=task_uuid,
                    message=str(e),
                    traceback=traceback.format_exc(),
                    elapsed_time=int(time.time() - start),
                )
        except Exception as db_error:
            logger.error(
                _get_log_message(
                    VacuumTaskPayload.NAME, task_id, f"Could not record failure to DB: {db_error}"
                )
            )


@celery_app.task(queue="cpu_queue", ignore_result=True)
def scan_library_task(task_id: str, payload: dict[str, Any]) -> None:
    """
    Scan the music library.
    """
    start = time.time()
    logger.info(_get_log_message(ScanTaskPayload.NAME, task_id, "Received task."))
    try:
        task_uuid = UUID(task_id)
        task_payload = ScanTaskPayload(**payload)
    except Exception as e:
        logger.error(
            _get_log_message(ScanTaskPayload.NAME, task_id, f"Failed to parse payload: {e}")
        )
        return

    try:
        with db_session_context() as db:
            if _skip_task_if_already_running(db, ScanTaskPayload.NAME, task_uuid, start):
                return

            task_manager.update_task_status(
                db,
                task_id=task_uuid,
                status=TaskStatus.RUNNING,
                elapsed_time=int(time.time() - start),
            )

            scan_path = Path(config.MUSIC_NAS_ROOT_DIR) / Path(task_payload.target_path)
            logger.info(
                _get_log_message(ScanTaskPayload.NAME, task_id, f"Started scan for: {scan_path}")
            )

            audio_files = list(audio.find_audio_files(scan_path))
            logger.info(
                _get_log_message(
                    ScanTaskPayload.NAME, task_id, f"Found {len(audio_files)} total audio files."
                )
            )

            if len(audio_files) == 0:
                task_manager.mark_task_as_success(
                    db,
                    task_id=task_uuid,
                    message=f"No audio files found to scan at {scan_path}.",
                    elapsed_time=int(time.time() - start),
                )
                return

            logger.info(
                _get_log_message(ScanTaskPayload.NAME, task_id, "Fetching existing file paths.")
            )
            existing_paths = set(db.exec(select(Song.file_path)).all())
            logger.info(
                _get_log_message(
                    ScanTaskPayload.NAME,
                    task_id,
                    f"Found {len(existing_paths)} existing files in database.",
                )
            )

            new_audio_files: list[Path] = []
            for audio_file in audio_files:
                relative_path = str(audio_file.relative_to(config.MUSIC_NAS_ROOT_DIR))
                if relative_path not in existing_paths:
                    new_audio_files.append(audio_file)

            total_files = len(new_audio_files)
            logger.info(
                _get_log_message(
                    ScanTaskPayload.NAME,
                    task_id,
                    f"Found {len(audio_files)} total files, {total_files} new files to process.",
                )
            )

            if total_files == 0:
                task_manager.mark_task_as_success(
                    db,
                    task_id=task_uuid,
                    message=f"No new audio files found to scan at {scan_path}.",
                    elapsed_time=int(time.time() - start),
                )
                return

            chunk_size = 100
            file_chunks = [
                new_audio_files[i : i + chunk_size] for i in range(0, total_files, chunk_size)
            ]
            processed_files = 0
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_chunk = {
                    executor.submit(audio.extract_metadata_bulk, chunk): chunk
                    for chunk in file_chunks
                }
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    metadata_chunk = future.result()

                    if metadata_chunk:
                        songs_to_insert = [
                            Song(
                                file_path=str(
                                    Path(m.file_path).relative_to(config.MUSIC_NAS_ROOT_DIR)
                                ),
                                title=m.title,
                                artist=m.artist,
                                album=m.album,
                                genre=m.genre,
                                duration_seconds=m.duration_seconds,
                                mtime=m.mtime,
                            )
                            for m in metadata_chunk
                        ]
                        db.add_all(songs_to_insert)
                        db.commit()

                    processed_files += len(chunk)
                    progress = int((processed_files / total_files) * 100)
                    task_manager.update_task_progress(
                        db,
                        task_id=task_uuid,
                        progress=progress,
                        elapsed_time=int(time.time() - start),
                    )

            task_manager.mark_task_as_success(
                db,
                task_id=task_uuid,
                message=f"Successfully scanned and processed {processed_files} files.",
                elapsed_time=int(time.time() - start),
            )

            logger.info(_get_log_message(ScanTaskPayload.NAME, task_id, "Successfully finished."))

    except Exception as e:
        logger.error(_get_log_message(ScanTaskPayload.NAME, task_id, f"Failed with error: {e}"))
        try:
            with db_session_context() as db:
                task_manager.mark_task_as_failure(
                    db,
                    task_id=task_uuid,
                    message=str(e),
                    traceback=traceback.format_exc(),
                    elapsed_time=int(time.time() - start),
                )
        except Exception as db_error:
            logger.error(
                _get_log_message(
                    ScanTaskPayload.NAME, task_id, f"Could not record failure to DB: {db_error}"
                )
            )


@celery_app.task(queue="cpu_queue", ignore_result=True)
def analyze_song_task(task_id: str, payload: dict[str, Any]) -> None:
    start = time.time()
    logger.info(_get_log_message(AnalyzeSongTaskPayload.NAME, task_id, "Received task."))
    try:
        task_uuid = UUID(task_id)
        task_payload = AnalyzeSongTaskPayload(**payload)
        models = [EmbeddingModel(m) for m in task_payload.models]
    except Exception as e:
        logging.error(
            _get_log_message(AnalyzeSongTaskPayload.NAME, task_id, f"Failed to parse payload: {e}")
        )
        return

    try:
        with db_session_context() as db:
            if _skip_task_if_already_running(db, AnalyzeSongTaskPayload.NAME, task_uuid, start):
                return

            target_songs: dict[EmbeddingModel, list[Song]] = {model: [] for model in models}

            if EmbeddingModel.ACOUSTIC_FEATURES in models:
                target_songs[EmbeddingModel.ACOUSTIC_FEATURES] = list(
                    db.exec(
                        select(Song).where(
                            ~select(SongEmbedding)
                            .where(SongEmbedding.file_path == Song.file_path)
                            .where(SongEmbedding.model_name == EmbeddingModel.ACOUSTIC_FEATURES)
                            .exists()
                        )
                    ).all()
                )

            embedding_model_types = {
                EmbeddingModel.CLAP,
                EmbeddingModel.MERT,
                EmbeddingModel.MUQ,
                EmbeddingModel.MUQ_MULAN,
            }
            for model_type in embedding_model_types:
                if model_type in models:
                    target_songs[model_type] = list(
                        db.exec(
                            select(Song).where(
                                ~select(SongEmbedding)
                                .where(SongEmbedding.file_path == Song.file_path)
                                .where(SongEmbedding.model_name == model_type)
                                .exists()
                            )
                        ).all()
                    )

            total_count = sum(len(songs) for songs in target_songs.values())

            if total_count == 0:
                task_manager.mark_task_as_success(
                    db,
                    task_id=task_uuid,
                    message="No songs found that need analysis or embeddings.",
                    elapsed_time=int(time.time() - start),
                )
                return

            logger.info(
                f"Found {total_count} tasks: "
                + ", ".join(
                    f"{model.value}({len(songs)})" for model, songs in target_songs.items() if songs
                )
            )

            task_groups: list[Signature] = []

            # Acoustic Features Chain
            if songs_for_acoustic := target_songs.get(EmbeddingModel.ACOUSTIC_FEATURES):
                extraction_task_group = group(
                    [
                        extract_acoustic_features.s(
                            task_id=task_id,
                            start_time=start,
                            file_paths=[s.file_path for s in chunk],
                        )
                        for chunk in (
                            songs_for_acoustic[i : i + TASK_CHUNK_SIZE]
                            for i in range(0, len(songs_for_acoustic), TASK_CHUNK_SIZE)
                        )
                    ]
                )
                vectorization_task_group = group(
                    [
                        vectorize_song_features.s(
                            task_id=task_id,
                            start_time=start,
                            file_paths=[s.file_path for s in chunk],
                        )
                        for chunk in (
                            songs_for_acoustic[i : i + VECTORIZE_CHUNK_SIZE]
                            for i in range(0, len(songs_for_acoustic), VECTORIZE_CHUNK_SIZE)
                        )
                    ]
                )
                task_groups.append(chain(extraction_task_group, vectorization_task_group))

            # Other Embedding Models Group
            embedding_tasks = []
            for model_type in embedding_model_types:
                if model_type not in models:
                    continue
                if songs_for_embedding := target_songs.get(model_type):
                    embedding_task_group = group(
                        [
                            generate_song_embedding.si(
                                task_id=task_id,
                                start_time=start,
                                file_paths=[s.file_path for s in chunk],
                                model_name=model_type.value,
                            )
                            for chunk in (
                                songs_for_embedding[i : i + TASK_CHUNK_SIZE]
                                for i in range(0, len(songs_for_embedding), TASK_CHUNK_SIZE)
                            )
                        ]
                    )
                    embedding_tasks.append(embedding_task_group)
            if embedding_tasks:
                task_groups.append(chain(*embedding_tasks))

            if not task_groups:
                task_manager.mark_task_as_success(
                    db,
                    task_id=task_uuid,
                    message="No tasks to run for the specified models.",
                    elapsed_time=int(time.time() - start),
                )
                return

            main_task_group = group(*task_groups)
            callback = finalize_analysis_task.s(task_id=task_id, start_time=start)
            task_manager.update_task_status(
                db,
                task_id=task_uuid,
                status=TaskStatus.RUNNING,
                details={"total_count": total_count, "processed_count": 0},
            )
            chord(main_task_group)(callback)
    except Exception as e:
        _handle_exception(AnalyzeSongTaskPayload.NAME, task_id, start, e)


@celery_app.task(queue="cpu_queue")
def extract_acoustic_features(task_id: str, start_time: float, file_paths: list[str]) -> bool:
    """
    Extracts acoustic features for a chunk of songs and saves them to the database.
    """
    processed_paths = []
    try:
        with db_session_context() as db:
            for file_path in file_paths:
                logger.info(f"Extracting features for {file_path}")
                features = acoustic_features.extract_acoustic_features(file_path)
                if features:
                    db.exec(delete(SongFeatures).where(SongFeatures.file_path == file_path))  # type: ignore
                    db.add(features)
                    processed_paths.append(file_path)
                    db.commit()
        record_progress(
            task_id=task_id, start_time=start_time, processed_count=len(processed_paths)
        )
        return True
    except Exception as e:
        _handle_exception("Extract acoustic features", task_id, start_time, e)
        return False


@celery_app.task(queue="cpu_queue")
def vectorize_song_features(
    _previous_results: list[Any], task_id: str, start_time: float, file_paths: list[str]
) -> bool:
    """Vectorize song features and upsert them to Milvus."""
    try:
        stats = acoustic_features.calculate_features_stats()

        with db_session_context() as db:
            songs_to_vectorize = db.exec(
                select(SongFeatures).where(col(SongFeatures.file_path).in_(file_paths))
            ).all()
            target_file_paths = [song.file_path for song in songs_to_vectorize]

            milvus_data = [
                acoustic_features.create_feature_vector_data(sf, stats) for sf in songs_to_vectorize
            ]
            new_embedding_records = [
                SongEmbedding(
                    file_path=file_path,
                    model_name=EmbeddingModel.ACOUSTIC_FEATURES,
                    status=Status.COMPLETED,
                    milvus_collection_name=acoustic_features.COLLECTION_NAME,
                    dimension=acoustic_features.EMBEDDING_DIMENSION,
                )
                for file_path in target_file_paths
            ]

            if milvus_data:
                acoustic_features.create_collection_if_not_exists()
                milvus_utils.upsert_vectors(acoustic_features.COLLECTION_NAME, milvus_data)
                logger.info(f"Successfully upserted {len(milvus_data)} vectors")

            if new_embedding_records:
                db.exec(
                    delete(SongEmbedding)
                    .where(col(SongEmbedding.file_path).in_(target_file_paths))
                    .where(col(SongEmbedding.model_name) == EmbeddingModel.ACOUSTIC_FEATURES)
                )  # type: ignore
                db.add_all(new_embedding_records)
                db.commit()

        record_progress(
            task_id=task_id, start_time=start_time, processed_count=len(target_file_paths)
        )
        return True
    except Exception as e:
        _handle_exception("Vectorize song features", task_id, start_time, e)
        return False


@celery_app.task(queue="cpu_queue")
def generate_song_embedding(
    task_id: str, start_time: float, file_paths: list[str], model_name: str
) -> bool:
    """
    Generates embeddings for a chunk of songs using the specified model,
    saves them to Milvus, and records metadata in PostgreSQL.
    """
    try:
        model_name_enum = EmbeddingModel(model_name)
    except ValueError as e:
        _handle_exception("Generate song embedding", task_id, start_time, e)
        return False

    milvus_data: list[dict[str, Any]] = []
    new_embedding_records: list[SongEmbedding] = []

    logger.info(f"Generating {model_name_enum.value} embedding for {file_paths}")
    full_audio_paths = [Path(config.MUSIC_NAS_ROOT_DIR) / file_path for file_path in file_paths]
    embeddings_dict = {}
    try:
        if model_name_enum == EmbeddingModel.CLAP:
            embeddings_dict = clap.get_audio_embeddings_batch(full_audio_paths)
            milvus_data.extend(
                [
                    {
                        clap.ID_FIELD: file_path,
                        clap.VECTOR_FIELD: embeddings_dict[
                            str(Path(config.MUSIC_NAS_ROOT_DIR) / file_path)
                        ],
                    }
                    for file_path in file_paths
                ]
            )
            new_embedding_records.extend(
                [
                    SongEmbedding(
                        file_path=file_path,
                        model_name=model_name_enum,
                        status=Status.COMPLETED,
                        milvus_collection_name=clap.COLLECTION_NAME,
                        dimension=clap.EMBEDDING_DIMENSION,
                    )
                    for file_path in file_paths
                ]
            )
        elif model_name_enum == EmbeddingModel.MERT:
            embeddings_dict = mert.get_audio_embeddings_batch(full_audio_paths)
            milvus_data.extend(
                [
                    {
                        mert.ID_FIELD: file_path,
                        mert.VECTOR_FIELD: embeddings_dict[
                            str(Path(config.MUSIC_NAS_ROOT_DIR) / file_path)
                        ],
                    }
                    for file_path in file_paths
                ]
            )
            new_embedding_records.extend(
                [
                    SongEmbedding(
                        file_path=file_path,
                        model_name=model_name_enum,
                        status=Status.COMPLETED,
                        milvus_collection_name=mert.COLLECTION_NAME,
                        dimension=mert.EMBEDDING_DIMENSION,
                    )
                    for file_path in file_paths
                ]
            )
        elif model_name_enum == EmbeddingModel.MUQ:
            embeddings_dict = muq.get_audio_embeddings_batch(full_audio_paths)
            milvus_data.extend(
                [
                    {
                        muq.ID_FIELD: file_path,
                        muq.VECTOR_FIELD: embeddings_dict[
                            str(Path(config.MUSIC_NAS_ROOT_DIR) / file_path)
                        ],
                    }
                    for file_path in file_paths
                ]
            )
            new_embedding_records.extend(
                [
                    SongEmbedding(
                        file_path=file_path,
                        model_name=model_name_enum,
                        status=Status.COMPLETED,
                        milvus_collection_name=muq.COLLECTION_NAME,
                        dimension=muq.EMBEDDING_DIMENSION,
                    )
                    for file_path in file_paths
                ]
            )
        elif model_name_enum == EmbeddingModel.MUQ_MULAN:
            embeddings_dict = muq_mulan.get_audio_embeddings_batch(full_audio_paths)
            milvus_data.extend(
                [
                    {
                        muq_mulan.ID_FIELD: file_path,
                        muq_mulan.VECTOR_FIELD: embeddings_dict[
                            str(Path(config.MUSIC_NAS_ROOT_DIR) / file_path)
                        ],
                    }
                    for file_path in file_paths
                ]
            )
            new_embedding_records.extend(
                [
                    SongEmbedding(
                        file_path=file_path,
                        model_name=model_name_enum,
                        status=Status.COMPLETED,
                        milvus_collection_name=muq_mulan.COLLECTION_NAME,
                        dimension=muq_mulan.EMBEDDING_DIMENSION,
                    )
                    for file_path in file_paths
                ]
            )
    except Exception as e:
        _handle_exception(
            "Generate song embedding",
            task_id,
            start_time,
            Exception(
                f"Failed to generate {model_name_enum.value} embedding for '{file_paths}': {e}"
            ),
        )
        return False

    try:
        if model_name_enum == EmbeddingModel.CLAP:
            clap.create_milvus_collection_if_not_exist()
            milvus_utils.upsert_vectors(clap.COLLECTION_NAME, milvus_data)
        elif model_name_enum == EmbeddingModel.MERT:
            mert.create_milvus_collection_if_not_exist()
            milvus_utils.upsert_vectors(mert.COLLECTION_NAME, milvus_data)
        elif model_name_enum == EmbeddingModel.MUQ:
            muq.create_milvus_collection_if_not_exist()
            milvus_utils.upsert_vectors(muq.COLLECTION_NAME, milvus_data)
        elif model_name_enum == EmbeddingModel.MUQ_MULAN:
            muq_mulan.create_milvus_collection_if_not_exist()
            milvus_utils.upsert_vectors(muq_mulan.COLLECTION_NAME, milvus_data)
        with db_session_context() as db:
            db.exec(
                delete(SongEmbedding)
                .where(col(SongEmbedding.file_path).in_(file_paths))
                .where(col(SongEmbedding.model_name) == model_name_enum)
            )  # type: ignore
            db.add_all(new_embedding_records)
            db.commit()
    except Exception as e:
        _handle_exception(
            "Generate song embedding",
            task_id,
            start_time,
            Exception(f"Failed to save embeddings for batch. Error: {e}"),
        )
        return False

    record_progress(task_id=task_id, start_time=start_time, processed_count=len(file_paths))
    return True


def _handle_exception(name: str, task_id: str, start: float, e: Exception) -> None:
    logger.error(_get_log_message(name, task_id, f"Failed with error: {e}"))
    try:
        with db_session_context() as db:
            task_manager.mark_task_as_failure(
                db,
                task_id=UUID(task_id),
                message=str(e),
                traceback=traceback.format_exc(),
                elapsed_time=int(time.time() - start),
                status=TaskStatus.RUNNING,  # Failed, but still running.
            )
    except Exception as db_error:
        logger.error(
            _get_log_message(
                name,
                task_id,
                f"Could not record failure to DB: {db_error}",
            )
        )


@celery_app.task(queue="cpu_queue", ignore_result=True)
def record_progress(task_id: str, start_time: float, processed_count: int) -> None:
    """Records the progress of a parent task in a transaction-safe way."""
    with db_session_context() as db:
        task = db.exec(
            select(TaskResult).where(TaskResult.id == task_id).with_for_update()
        ).one_or_none()

        if not task or not task.details:
            logger.warning(f"Could not find task {task_id} to record progress, or details not set.")
            return

        current_processed = task.details.get("processed_count", 0) + processed_count
        total_count = task.details.get("total_count", 0)

        if current_processed == total_count:
            task_manager.mark_task_as_success(
                db,
                task_id=UUID(task_id),
                message="Analysis and embedding generation completed successfully.",
                elapsed_time=int(time.time() - start_time),
                details={"processed_count": current_processed, "total_count": total_count},
            )
            return

        new_progress = int((current_processed / total_count) * 100) if total_count > 0 else 0

        task.progress = new_progress
        task.details = {"processed_count": current_processed, "total_count": total_count}
        task.elapsed_time = int(time.time() - start_time)

        db.commit()
        db.refresh(task)


@celery_app.task(queue="cpu_queue", ignore_result=True)
def finalize_analysis_task(_results: list[Any], task_id: str, start_time: float) -> None:
    """
    Callback task that runs after all analysis and embedding sub-tasks are complete.
    It marks the main task as successful.
    """
    with db_session_context() as db:
        task = db.exec(
            select(TaskResult).where(TaskResult.id == task_id).with_for_update()
        ).one_or_none()

        if not task:
            logger.warning(f"Could not find task {task_id} to finalize.")
            return

        if task.traceback:
            task_manager.mark_task_as_failure(
                db,
                task_id=UUID(task_id),
                message="Analysis and embedding generation failed.",
                elapsed_time=int(time.time() - start_time),
                traceback=task.traceback,
                details=task.details,
            )
            return
        task_manager.mark_task_as_success(
            db,
            task_id=UUID(task_id),
            message="Analysis and embedding generation completed successfully.",
            elapsed_time=int(time.time() - start_time),
            details=task.details,
        )
    logger.info(
        _get_log_message(AnalyzeSongTaskPayload.NAME, str(task_id), "Chord finished successfully.")
    )
