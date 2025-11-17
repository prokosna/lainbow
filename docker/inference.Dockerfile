FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /src

COPY ./docker/inference_requirements.txt /src/requirements.txt
COPY ./docker/batch_requirements.txt /src/batch_requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r batch_requirements.txt

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

EXPOSE 8001

CMD ["uvicorn", "inference.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
