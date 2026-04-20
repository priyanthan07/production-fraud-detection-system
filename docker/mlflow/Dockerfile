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
    "protobuf>=3.20.0,<4.0.0" \
    "sqlalchemy==1.4.46"

# Create artifact storage directory
RUN mkdir -p /mlflow/artifacts

# Expose MLflow UI port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

CMD ["mlflow", "server", "--backend-store-uri", "postgresql://postgres:admin@postgres:5432/mlflow_tracking", "--default-artifact-root", "/mlflow/artifacts", "--host", "0.0.0.0", "--port", "5000"]
