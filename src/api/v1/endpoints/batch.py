from uuid import UUID

from api.session_manager_wrap import get_db_session
from domain import task_manager
from domain.messages import AnalyzeSongTaskPayload, ScanTaskPayload, VacuumTaskPayload
from domain.schemas import EmbeddingModel, TaskResult
from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session
from worker.tasks import analyze_song_task, scan_library_task, vacuum_library_task

router = APIRouter()


@router.post(
    "/scan/run",
    response_model=TaskResult,
    summary="Start a music library scan",
    description="Triggers a task to scan the specified path and records states of all files under the path in the database.",
)
def run_scan(
    db: Session = Depends(get_db_session),  # noqa: B008
) -> TaskResult:
    """Enqueues a task to scan the music library and records it in the database."""
    try:
        db_task = task_manager.create_task(db, name=str(ScanTaskPayload.NAME))
        payload = ScanTaskPayload(task_id=db_task.id, target_path=".")
        scan_library_task.delay(task_id=str(db_task.id), payload=payload.model_dump())
        return db_task

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to enqueue scan task. Error: {e}",
        ) from e


@router.get(
    "/tasks/{task_id}",
    response_model=TaskResult,
    summary="Get task status",
    description="Retrieves the current status and result of a specific background task.",
)
def get_task_status(
    task_id: UUID,
    db: Session = Depends(get_db_session),  # noqa: B008
) -> TaskResult:
    """Retrieves the status of a specific task from the database."""
    db_task = task_manager.get_task(db, task_id=task_id)
    if not db_task:
        raise HTTPException(status_code=404, detail="Task not found")
    return db_task


@router.post("/vacuum/run", response_model=TaskResult, summary="Run library vacuum")
def run_vacuum(
    db: Session = Depends(get_db_session),  # noqa: B008
) -> TaskResult:
    """Enqueue a new task to vacuum the music library."""
    try:
        db_task = task_manager.create_task(db, name=VacuumTaskPayload.NAME)
        db.commit()
        db.refresh(db_task)

        task_payload = VacuumTaskPayload(task_id=db_task.id)

        vacuum_library_task.delay(task_id=str(db_task.id), payload=task_payload.model_dump())
        return db_task
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enqueue vacuum task: {e}") from e


class RunAnalysisRequest(BaseModel):
    models: list[str] = Field(
        default_factory=lambda: [EmbeddingModel.MUQ.value, EmbeddingModel.MUQ_MULAN.value]
    )


@router.post("/analyze/run", response_model=TaskResult, summary="Run song analysis")
def run_analysis(
    db: Session = Depends(get_db_session),  # noqa: B008
    request: RunAnalysisRequest = Body(default_factory=RunAnalysisRequest),  # noqa: B008
) -> TaskResult:
    """
    Enqueue a task to analyze all songs and generate embeddings.
    """
    try:
        db_task = task_manager.create_task(db, name=AnalyzeSongTaskPayload.NAME)
        db.commit()
        db.refresh(db_task)

        task_payload = AnalyzeSongTaskPayload(task_id=db_task.id, models=request.models)

        analyze_song_task.delay(task_id=str(db_task.id), payload=task_payload.model_dump())
        return db_task
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enqueue analysis task: {e}") from e
