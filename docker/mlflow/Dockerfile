FROM python:3.12-slim

# Install PostgreSQL client library (needed by psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /mlflow

# Install MLflow and PostgreSQL adapter
# Pin versions to match what the project uses
RUN pip install --no-cache-dir \
    "mlflow>=2.10.0,<3.0.0" \
    "psycopg2-binary>=2.9.0" \
    "sqlalchemy==1.4.46"

# Create artifact storage directory
RUN mkdir -p /mlflow/artifacts

# Expose MLflow UI port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# --serve-artifacts   → MLflow proxies artifact uploads/downloads through the
#                       HTTP server instead of returning raw filesystem paths.
#                       Without this flag, clients receive file:///mlflow/artifacts/...
#                       URIs and try to write directly to that path on their own disk,
#                       which fails in CI (no /mlflow directory on the runner) and
#                       breaks the inference container (path not accessible).
#
# --artifacts-destination → Where the server stores artifact files.
#                           Must match the Docker volume mount point.
CMD ["mlflow", "server", \
    "--backend-store-uri", "postgresql://postgres:admin@postgres:5432/mlflow_tracking", \
    "--artifacts-destination", "/mlflow/artifacts", \
    "--serve-artifacts", \
    "--host", "0.0.0.0", \
    "--port", "5000"]