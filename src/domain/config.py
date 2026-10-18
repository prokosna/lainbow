import os

from domain.enums import EmbeddingModel

# Database server endpoint (legacy fallback for all datastores)
DATABASE_SERVER_ENDPOINT = os.getenv("DATABASE_SERVER_ENDPOINT", "host.docker.internal")

# PostgreSQL
# POSTGRES_PORT is the host-published port (used by Docker Compose).
# POSTGRES_HOST and POSTGRES_CONNECT_PORT are used by application containers to connect.
POSTGRES_USER = os.getenv("POSTGRES_USER", "user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
POSTGRES_DB = os.getenv("POSTGRES_DB", "lainbow_db")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5433")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", DATABASE_SERVER_ENDPOINT)
POSTGRES_CONNECT_PORT = os.getenv("POSTGRES_CONNECT_PORT", POSTGRES_PORT)
POSTGRES_URL = f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_CONNECT_PORT}/{POSTGRES_DB}"
CELERY_BACKEND_URL = f"db+postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_CONNECT_PORT}/{POSTGRES_DB}"

# RabbitMQ
# RABBITMQ_PORT is the host-published port (used by Docker Compose).
# RABBITMQ_HOST and RABBITMQ_CONNECT_PORT are used by application containers to connect.
RABBITMQ_USER = os.getenv("RABBITMQ_DEFAULT_USER", "user")
RABBITMQ_PASS = os.getenv("RABBITMQ_DEFAULT_PASS", "password")
RABBITMQ_PORT = os.getenv("RABBITMQ_PORT", "5672")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", DATABASE_SERVER_ENDPOINT)
RABBITMQ_CONNECT_PORT = os.getenv("RABBITMQ_CONNECT_PORT", RABBITMQ_PORT)
RABBITMQ_URL = f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASS}@{RABBITMQ_HOST}:{RABBITMQ_CONNECT_PORT}/"

# Qdrant
# QDRANT_PORT (from QDRANT_PORT_HTTP) is the host-published port (used by Docker Compose).
# QDRANT_HOST and QDRANT_CONNECT_PORT are used by application clients to connect.
QDRANT_PORT = int(os.getenv("QDRANT_PORT_HTTP", "6333"))
QDRANT_HOST = os.getenv("QDRANT_HOST", DATABASE_SERVER_ENDPOINT)
QDRANT_CONNECT_PORT = int(os.getenv("QDRANT_CONNECT_PORT", str(QDRANT_PORT)))

# Music NAS folder path (in the container - you should not change this)
MUSIC_NAS_ROOT_DIR = "/music_library"

# Inference API
INFERENCE_SERVER_ENDPOINT = os.getenv("INFERENCE_SERVER_ENDPOINT", "host.docker.internal")
INFERENCE_API_PORT = int(os.getenv("INFERENCE_API_PORT", "8001"))
INFERENCE_MODEL_MANAGER_MODE = os.getenv("INFERENCE_MODEL_MANAGER_MODE", "single")
INFERENCE_API_URL = os.getenv(
    "INFERENCE_API_URL", f"http://{INFERENCE_SERVER_ENDPOINT}:{INFERENCE_API_PORT}"
)
INFERENCE_REQUEST_TIMEOUT = float(os.getenv("INFERENCE_REQUEST_TIMEOUT", "300.0"))
INFERENCE_TEXT_TIMEOUT = float(os.getenv("INFERENCE_TEXT_TIMEOUT", "30.0"))
INFERENCE_LOAD_MODELS_ON_STARTUP = [
    model.strip()
    for model in os.getenv("INFERENCE_MODELS_TO_LOAD_ON_STARTUP", "").split(",")
    if model.strip()
]
INFERENCE_PINNED_MODELS = [
    model.strip()
    for model in os.getenv("INFERENCE_PINNED_MODELS", "muq_mulan").split(",")
    if model.strip()
]

# Batch
TASK_CHUNK_SIZE = int(os.getenv("TASK_CHUNK_SIZE", "10"))
MUQ_FILE_DURATION_SEC = int(os.getenv("MUQ_FILE_DURATION_SEC", "180"))
TASK_STALE_AFTER_SEC = int(os.getenv("TASK_STALE_AFTER_SEC", "1800"))


def _parse_analyze_models(raw: str) -> list[str]:
    valid_names = {model.value for model in EmbeddingModel}
    models = [m.strip() for m in raw.split(",") if m.strip()]
    for m in models:
        if m not in valid_names:
            raise ValueError(
                f"Unknown model '{m}' in LAINBOW_ANALYZE_MODELS. "
                f"Valid choices are: {', '.join(sorted(valid_names))}"
            )
    return models


ANALYZE_MODELS = _parse_analyze_models(os.getenv("LAINBOW_ANALYZE_MODELS", "muq,muq_mulan"))
