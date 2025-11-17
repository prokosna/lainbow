FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /src

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY ./docker/inference_requirements.txt /src/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8001

CMD ["uvicorn", "inference.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
