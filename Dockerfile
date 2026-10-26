FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-export.txt .
RUN pip install --no-cache-dir -r requirements-export.txt
COPY app ./app
ENV DATASET_ROOT=/dataset RUNS_DIR=/runs PORT=8768
EXPOSE 8768
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8768}"]
