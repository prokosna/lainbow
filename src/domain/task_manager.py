from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from domain.schemas import TaskResult, TaskStatus


def create_task(
    db: Session,
    name: str,
) -> TaskResult:
    """Creates a new task record in the database."""
    new_task = TaskResult(name=name, status=TaskStatus.PENDING)
    db.add(new_task)
    db.commit()
    db.refresh(new_task)
    return new_task


def get_task(db: Session, task_id: UUID) -> TaskResult | None:
    """Retrieves a task from the database by its ID."""
    result = db.exec(select(TaskResult).where(TaskResult.id == task_id))
    return result.one_or_none()


def get_latest_task_by_name(db: Session, name: str, excluding_id: UUID) -> TaskResult | None:
    """Retrieves the latest task from the database by its name."""
    result = db.exec(
        select(TaskResult)
        .where(TaskResult.name == name)
        .where(TaskResult.id != excluding_id)
        .order_by(TaskResult.updated_at.desc())  # type: ignore
    )
    return result.first()


def update_task_status(
    db: Session,
    task_id: UUID,
    status: TaskStatus,
    elapsed_time: int | None = None,
    details: dict[str, Any] | None = None,
) -> TaskResult | None:
    """Updates the status of a task."""
    task = get_task(db, task_id)
    if task:
        task.status = status
        task.elapsed_time = elapsed_time
        task.details = details
        db.commit()
        db.refresh(task)
    return task


def update_task_progress(
    db: Session,
    task_id: UUID,
    progress: int,
    elapsed_time: int | None = None,
    details: dict[str, Any] | None = None,
) -> TaskResult | None:
    """Updates the progress percentage of a task."""
    task = get_task(db, task_id)
    if task:
        task.progress = progress
        task.elapsed_time = elapsed_time
        task.details = details
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.RUNNING
        db.commit()
        db.refresh(task)
    return task


def mark_task_as_success(
    db: Session,
    task_id: UUID,
    message: str | None = None,
    elapsed_time: int | None = None,
    traceback: str | None = None,
    details: dict[str, Any] | None = None,
) -> TaskResult | None:
    """Marks a task as successful and records its result."""
    task = get_task(db, task_id)
    if task:
        task.status = TaskStatus.SUCCESS
        task.message = message
        task.elapsed_time = elapsed_time
        task.progress = 100
        task.traceback = traceback if traceback is not None else task.traceback
        task.details = details if details is not None else task.details
        db.commit()
        db.refresh(task)
    return task


def mark_task_as_failure(
    db: Session,
    task_id: UUID,
    message: str | None = None,
    elapsed_time: int | None = None,
    traceback: str | None = None,
    details: dict[str, Any] | None = None,
    status: TaskStatus = TaskStatus.FAILURE,
) -> TaskResult | None:
    """Marks a task as failed and records the error details."""
    task = get_task(db, task_id)
    if task:
        task.status = status
        task.message = message
        task.elapsed_time = elapsed_time
        task.traceback = traceback
        task.details = details if details is not None else task.details
        db.commit()
        db.refresh(task)
    return task
