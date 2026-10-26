FROM python:3.10-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 ACTIVE_BUNDLE=bundles/nii-small-v1 UPLOAD_DIR=/tmp/uploads TMPDIR=/tmp/uploads DONATION_DIR=/data/donations ONNX_THREADS=4 ANALYSIS_CONCURRENCY=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /service
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY app ./app
COPY pipeline ./pipeline
COPY donations ./donations
COPY static ./static
COPY bundles ./bundles
COPY models ./models
COPY tests ./tests
COPY scripts ./scripts
COPY calibration/__init__.py calibration/fit_nii_v1.py ./calibration/
COPY RUNTIME_AUDIT.json ./REUSE_AUDIT.json
COPY pytest.ini ./
COPY MODEL_ATTRIBUTION.md ./
RUN useradd --uid 10001 --create-home service && mkdir -p /tmp/uploads /data/donations /service/work && chown -R service:service /tmp/uploads /data/donations /service/work
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=6 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=3)"
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--log-config", "scripts/logging.json"]
