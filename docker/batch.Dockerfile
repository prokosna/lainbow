FROM python:3.12-slim

WORKDIR /src

COPY ./docker/batch_requirements.txt /src/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

CMD ["celery", "-A", "worker.tasks", "worker", "--loglevel=info"]
