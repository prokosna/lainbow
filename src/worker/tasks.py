import datetime
import logging
import time
import traceback
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from uuid import UUID

from celery import chain, chord, group
from celery.canvas import Signature
from sqlmodel import Session, col, delete, or_, select, update

from domain import config, task_manager, vector_store_utils
from domain.embedding_models import INFERENCE_MODELS, MODEL_SPECS
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
from worker import acoustic_features, audio, embeddings
from worker.celery_app import celery_app

logger = logging.getLogger(__name__)

TASK_CHUNK_SIZE = config.TASK_CHUNK_SIZE


def _chunks[T](seq: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _get_log_message(name: str, task_id: str, message: str) -> str:
    return f"[{name} {task_id}] {message}"


def _skip_task_if_already_running(db: Session, name: str, task_id: UUID, start: float) -> bool:
    existing_task = task_manager.get_latest_task_by_name(db, name=name, excluding_id=task_id)
    if existing_task is not None and existing_task.status in [
        TaskStatus.RUNNING,
        TaskStatus.PENDING,
    ]:
        updated_at = existing_task.updated_at or existing_task.created_at
        if updated_at is not None:
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=datetime.UTC)
            age = (datetime.datetime.now(datetime.UTC) - updated_at).total_seconds()
            if age > config.TASK_STALE_AFTER_SEC:
                task_manager.mark_task_as_failure(
                    db,
                    task_id=existing_task.id,
                    message=f"Marked as failed: no update for over {config.TASK_STALE_AFTER_SEC} seconds (stale).",
                )
                return False

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
            skipped_count = 0
            song_infos = [
                (Path(config.MUSIC_NAS_ROOT_DIR) / s.file_path, s.mtime) for s in all_songs
            ]

            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_info = {
                    executor.submit(audio.check_audio_file_status, info[0], info[1]): info
                    for info in song_infos
                }

                for future in as_completed(future_to_info):
                    processed_songs += 1
                    try:
                        result = future.result()
                    except Exception as e:
                        info = future_to_info[future]
                        logger.warning(
                            _get_log_message(
                                VacuumTaskPayload.NAME,
                                task_id,
                                f"Failed to check audio file status for {info[0]}: {e}",
                            )
                        )
                        skipped_count += 1
                    else:
                        if result.status == "NOT_EXIST":
                            file_path = str(
                                Path(result.file_path).relative_to(config.MUSIC_NAS_ROOT_DIR)
                            )
                            logger.info(f"'{file_path}' does not exist. Deleting metadata.")

                            # Delete from PostgreSQL
                            db.exec(
                                delete(SongFeatures).where(col(SongFeatures.file_path) == file_path)
                            )
                            db.exec(
                                delete(SongEmbedding).where(
                                    col(SongEmbedding.file_path) == file_path
                                )
                            )
                            db.exec(delete(Song).where(col(Song.file_path) == file_path))
                            deleted_count += 1

                            # Delete from vector store
                            for spec in MODEL_SPECS.values():
                                embeddings.delete_vectors(spec, file_paths=[file_path])
                        elif result.status == "STALE":
                            file_path = str(
                                Path(result.file_path).relative_to(config.MUSIC_NAS_ROOT_DIR)
                            )

                            assert result.metadata is not None, (
                                "Metadata should not be None for STALE status"
                            )

                            logger.info(
                                f"'{file_path}' is stale. Updating metadata and deleting embeddings."
                            )

                            # Delete related analysis and embeddings from PostgreSQL to mark for re-calculation
                            db.exec(
                                delete(SongFeatures).where(col(SongFeatures.file_path) == file_path)
                            )
                            db.exec(
                                delete(SongEmbedding).where(
                                    col(SongEmbedding.file_path) == file_path
                                )
                            )

                            # Delete from vector store
                            for spec in MODEL_SPECS.values():
                                embeddings.delete_vectors(spec, file_paths=[file_path])

                            # Then, update the song's metadata
                            db.exec(
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
                        elif result.status == "UNREADABLE":
                            file_path = str(
                                Path(result.file_path).relative_to(config.MUSIC_NAS_ROOT_DIR)
                            )
                            logger.warning(
                                f"'{file_path}' is unreadable. Leaving DB/vector store untouched."
                            )
                            skipped_count += 1

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
                    f"Deleted {deleted_count} songs, Updated {updated_count} songs, Skipped {skipped_count} songs.",
                )
            )

            task_manager.mark_task_as_success(
                db,
                task_id=task_uuid,
                message=f"Vacuum complete. Deleted: {deleted_count}, Updated: {updated_count}, Skipped (unreadable): {skipped_count}.",
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
            successful_count = 0
            skipped_count = 0
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_chunk = {
                    executor.submit(audio.extract_metadata_bulk, chunk): chunk
                    for chunk in file_chunks
                }
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    try:
                        metadata_chunk = future.result()
                    except Exception as e:
                        logger.warning(
                            _get_log_message(
                                ScanTaskPayload.NAME,
                                task_id,
                                f"Failed to extract metadata chunk: {e}",
                            )
                        )
                        skipped_count += len(chunk)
                    else:
                        chunk_skipped = len(chunk) - len(metadata_chunk)
                        skipped_count += chunk_skipped
                        successful_count += len(metadata_chunk)

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
                message=f"Successfully scanned and processed {successful_count} files. Skipped (unreadable): {skipped_count}.",
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


def _select_songs_for_model(
    db: Session, model: EmbeddingModel | str, retry_failed: bool = False
) -> list[Song]:
    """Select songs that need analysis or embedding for a model."""
    model_enum = EmbeddingModel(model) if isinstance(model, str) else model
    if retry_failed:
        has_no_row = (
            ~select(SongEmbedding)
            .where(
                SongEmbedding.file_path == Song.file_path,
                SongEmbedding.model_name == model_enum,
            )
            .exists()
        )
        has_failed_row = (
            select(SongEmbedding)
            .where(
                SongEmbedding.file_path == Song.file_path,
                SongEmbedding.model_name == model_enum,
                SongEmbedding.status == Status.FAILED,
            )
            .exists()
        )
        stmt = select(Song).where(or_(has_no_row, has_failed_row))
    else:
        stmt = select(Song).where(
            ~select(SongEmbedding)
            .where(SongEmbedding.file_path == Song.file_path)
            .where(SongEmbedding.model_name == model_enum)
            .exists()
        )
    return list(db.exec(stmt).all())


@celery_app.task(queue="cpu_queue", ignore_result=True)
def analyze_song_task(task_id: str, payload: dict[str, Any]) -> None:
    """Analyze songs and generate embeddings for requested models.

    total_count counts one unit per song per requested model.
    """
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
                target_songs[EmbeddingModel.ACOUSTIC_FEATURES] = _select_songs_for_model(
                    db, EmbeddingModel.ACOUSTIC_FEATURES, retry_failed=task_payload.retry_failed
                )

            for model_type in INFERENCE_MODELS:
                if model_type in models:
                    target_songs[model_type] = _select_songs_for_model(
                        db, model_type, retry_failed=task_payload.retry_failed
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
                vectorization_task = vectorize_song_features.s(
                    task_id=task_id,
                    start_time=start,
                    file_paths=[s.file_path for s in songs_for_acoustic],
                )
                task_groups.append(chain(extraction_task_group, vectorization_task))

            # Other Embedding Models Group
            embedding_tasks = []
            for model_type in INFERENCE_MODELS:
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

            callback = finalize_analysis_task.s(task_id=task_id, start_time=start)
            task_manager.update_task_status(
                db,
                task_id=task_uuid,
                status=TaskStatus.RUNNING,
                details={"total_count": total_count, "processed_count": 0},
            )
            chord(task_groups)(callback)
    except Exception as e:
        _handle_exception(AnalyzeSongTaskPayload.NAME, task_id, start, e, status=TaskStatus.FAILURE)


@celery_app.task(queue="cpu_queue")
def extract_acoustic_features(task_id: str, start_time: float, file_paths: list[str]) -> bool:
    """
    Extracts acoustic features for a chunk of songs and saves them to the database.
    """
    try:
        with db_session_context() as db:
            for file_path in file_paths:
                logger.info(f"Extracting features for {file_path}")
                features = acoustic_features.extract_acoustic_features(file_path)
                if features:
                    db.exec(delete(SongFeatures).where(SongFeatures.file_path == file_path))  # type: ignore
                    db.add(features)
                    db.commit()
        return True
    except Exception as e:
        _handle_exception("Extract acoustic features", task_id, start_time, e)
        return False


@celery_app.task(queue="cpu_queue")
def vectorize_song_features(
    _previous_results: list[Any], task_id: str, start_time: float, file_paths: list[str]
) -> bool:
    """Vectorize song features and upsert them to the vector store."""
    try:
        stats = acoustic_features.calculate_features_stats()

        with db_session_context() as db:
            acoustic_features.create_collection_if_not_exists()

            page_size = 1000
            offset = 0
            while True:
                page = list(
                    db.exec(
                        select(SongFeatures)
                        .where(col(SongFeatures.status) == Status.COMPLETED)
                        .order_by(SongFeatures.file_path)
                        .offset(offset)
                        .limit(page_size)
                    ).all()
                )
                if not page:
                    break
                vector_store_data = [
                    acoustic_features.create_feature_vector_data(sf, stats) for sf in page
                ]
                vector_store_utils.upsert_vectors(
                    acoustic_features.COLLECTION_NAME, vector_store_data
                )
                offset += len(page)
                if len(page) < page_size:
                    break

            if file_paths:
                for chunk in _chunks(file_paths, 1000):
                    target_features = db.exec(
                        select(SongFeatures).where(col(SongFeatures.file_path).in_(chunk))
                    ).all()
                    features_by_path = {
                        song.file_path: song
                        for song in target_features
                        if song.status == Status.COMPLETED
                    }

                    new_embedding_records: list[SongEmbedding] = []
                    for file_path in chunk:
                        if file_path in features_by_path:
                            new_embedding_records.append(
                                SongEmbedding(
                                    file_path=file_path,
                                    model_name=EmbeddingModel.ACOUSTIC_FEATURES,
                                    status=Status.COMPLETED,
                                    vector_collection_name=acoustic_features.COLLECTION_NAME,
                                    dimension=acoustic_features.EMBEDDING_DIMENSION,
                                )
                            )
                        else:
                            logger.warning(
                                _get_log_message(
                                    "Vectorize song features",
                                    task_id,
                                    f"Missing SongFeatures for '{file_path}', marking as FAILED.",
                                )
                            )
                            new_embedding_records.append(
                                SongEmbedding(
                                    file_path=file_path,
                                    model_name=EmbeddingModel.ACOUSTIC_FEATURES,
                                    status=Status.FAILED,
                                    vector_collection_name=acoustic_features.COLLECTION_NAME,
                                    dimension=None,
                                )
                            )

                    if new_embedding_records:
                        db.exec(
                            delete(SongEmbedding)
                            .where(col(SongEmbedding.file_path).in_(chunk))
                            .where(
                                col(SongEmbedding.model_name) == EmbeddingModel.ACOUSTIC_FEATURES
                            )
                        )
                        db.add_all(new_embedding_records)
                        db.commit()

        record_progress(task_id=task_id, start_time=start_time, processed_count=len(file_paths))
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
    saves them to the vector store, and records metadata in PostgreSQL.
    """
    try:
        model_name_enum = EmbeddingModel(model_name)
        if (
            model_name_enum == EmbeddingModel.ACOUSTIC_FEATURES
            or model_name_enum not in MODEL_SPECS
        ):
            raise ValueError(f"Unsupported model: {model_name}")
        spec = MODEL_SPECS[model_name_enum]
    except ValueError as e:
        _handle_exception("Generate song embedding", task_id, start_time, e)
        return False

    vector_store_data: list[dict[str, Any]] = []
    new_embedding_records: list[SongEmbedding] = []

    logger.info(f"Generating {model_name_enum.value} embedding for {file_paths}")
    full_audio_paths = [Path(config.MUSIC_NAS_ROOT_DIR) / file_path for file_path in file_paths]
    try:
        embeddings_dict = embeddings.get_audio_embeddings_batch(spec, full_audio_paths)
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

    for file_path in file_paths:
        full_audio_path_str = str(Path(config.MUSIC_NAS_ROOT_DIR) / file_path)
        if full_audio_path_str in embeddings_dict:
            vector_store_data.append(
                {
                    "file_path": file_path,
                    "embedding": embeddings_dict[full_audio_path_str],
                }
            )
            new_embedding_records.append(
                SongEmbedding(
                    file_path=file_path,
                    model_name=model_name_enum,
                    status=Status.COMPLETED,
                    vector_collection_name=spec.collection_name,
                    dimension=spec.dimension,
                )
            )
        else:
            logger.warning(
                _get_log_message(
                    "Generate song embedding",
                    task_id,
                    f"Failed to generate {model_name_enum.value} embedding for '{file_path}' (omitted from batch result).",
                )
            )
            new_embedding_records.append(
                SongEmbedding(
                    file_path=file_path,
                    model_name=model_name_enum,
                    status=Status.FAILED,
                    vector_collection_name=spec.collection_name,
                    dimension=None,
                )
            )

    try:
        if vector_store_data:
            embeddings.ensure_vector_collection_ready(spec)
            vector_store_utils.upsert_vectors(spec.collection_name, vector_store_data)
        with db_session_context() as db:
            db.exec(
                delete(SongEmbedding)
                .where(col(SongEmbedding.file_path).in_(file_paths))
                .where(col(SongEmbedding.model_name) == model_name_enum)
            )
            db.add_all(new_embedding_records)
            db.commit()
        record_progress(task_id=task_id, start_time=start_time, processed_count=len(file_paths))
        return True
    except Exception as e:
        _handle_exception(
            "Generate song embedding",
            task_id,
            start_time,
            Exception(f"Failed to save embeddings for batch. Error: {e}"),
        )
        return False


def _handle_exception(
    name: str,
    task_id: str,
    start: float,
    e: Exception,
    status: TaskStatus = TaskStatus.RUNNING,
) -> None:
    logger.error(_get_log_message(name, task_id, f"Failed with error: {e}"))
    try:
        with db_session_context() as db:
            task_manager.mark_task_as_failure(
                db,
                task_id=UUID(task_id),
                message=str(e),
                traceback=traceback.format_exc(),
                elapsed_time=int(time.time() - start),
                status=status,
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
            select(TaskResult).where(TaskResult.id == UUID(task_id)).with_for_update()
        ).one_or_none()

        if not task or not task.details:
            logger.warning(f"Could not find task {task_id} to record progress, or details not set.")
            return

        if task.status == TaskStatus.SUCCESS:
            return

        current_processed = task.details.get("processed_count", 0) + processed_count
        total_count = task.details.get("total_count", 0)

        if current_processed >= total_count:
            task_manager.mark_task_as_success(
                db,
                task_id=UUID(task_id),
                message="Analysis and embedding generation completed successfully.",
                elapsed_time=int(time.time() - start_time),
                details={"processed_count": current_processed, "total_count": total_count},
            )
            return

        new_progress = (
            min(100, int((current_processed / total_count) * 100)) if total_count > 0 else 0
        )

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
            select(TaskResult).where(TaskResult.id == UUID(task_id)).with_for_update()
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
