FROM python:3.12-slim

WORKDIR /src

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.17 /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY ./pyproject.toml ./uv.lock /src/
RUN uv sync --locked --no-dev --group batch

COPY src /src

CMD ["celery", "-A", "worker.tasks", "worker", "--loglevel=info"]
