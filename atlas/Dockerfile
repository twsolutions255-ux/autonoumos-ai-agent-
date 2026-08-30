FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY atlas ./atlas
COPY scripts ./scripts

# Atlas holds superadmin credentials for the app it runs; it has no reason to
# be root inside its own container.
RUN useradd --create-home --shell /bin/false atlas && chown -R atlas:atlas /app
USER atlas

EXPOSE 8090

# One worker, deliberately. The scheduler runs inside the process, and a second
# worker would mean two agents running the same business on the same timer.
CMD ["sh", "-c", "uvicorn atlas.main:app --host 0.0.0.0 --port ${PORT:-8090} --workers 1"]
