import faulthandler
from typing import Any

from celery import Celery
from celery.signals import worker_process_init
from domain import config
from domain.session_manager import init_db_session
from kombu import Exchange, Queue

faulthandler.enable()


@worker_process_init.connect  # type: ignore[misc]
def on_worker_init(**kwargs: dict[str, Any]) -> None:
    """Initialize database connection for the worker process."""
    init_db_session()


celery_app = Celery(
    "lainbow_worker",
    broker=config.RABBITMQ_URL,
    backend=config.CELERY_BACKEND_URL,
    include=["worker.tasks"],
)

celery_app.conf.update(
    task_track_started=True,
    task_acks_late=True,
    task_default_priority=1,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    visibility_timeout=1800,
    task_queues=(
        Queue(
            "cpu_queue",
            Exchange("default"),
            routing_key="cpu_queue",
            queue_arguments={"x-max-priority": 10},
        ),
    ),
)
