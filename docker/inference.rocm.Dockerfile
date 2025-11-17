FROM rocm/pytorch:rocm7.1_ubuntu24.04_py3.12_pytorch_release_2.7.1

WORKDIR /src

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential clang \
    && rm -rf /var/lib/apt/lists/*

COPY ./docker/inference_requirements.txt /src/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8001

CMD ["uvicorn", "inference.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
