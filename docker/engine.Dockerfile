# Lainbow engine image: torch, the audio models and ffmpeg.
# ACCEL selects the torch build: cpu, cuda or rocm. The ROCm wheel brings its own runtime.
ARG ACCEL=cpu

FROM python:3.12-slim AS build
ARG ACCEL
COPY --from=ghcr.io/astral-sh/uv:0.9.17 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
WORKDIR /app

# Dependency layer: rebuilt only when pyproject.toml or uv.lock change.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-dev --no-editable --no-install-project \
        --extra engine --extra models --extra optional-models --extra ${ACCEL}

# Project layer: install lainbow itself (not editable, so src/ is not needed at runtime).
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable \
        --extra engine --extra models --extra optional-models --extra ${ACCEL}

FROM python:3.12-slim
# ffmpeg decodes audio; libsndfile1 and libgomp1 are runtime libraries of the audio and torch wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/venv /opt/venv
# The engine runs as root: the models directory is a bind mount that Docker creates root-owned.
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf
EXPOSE 8001
CMD ["lainbow-engine"]
