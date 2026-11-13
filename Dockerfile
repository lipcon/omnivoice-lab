# Torch's cu128 wheels bundle the CUDA runtime, so a CUDA base image is not
# needed -- only the host driver plus nvidia-container-toolkit.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.8.0 torchaudio==2.8.0

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY app /app
WORKDIR /app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
