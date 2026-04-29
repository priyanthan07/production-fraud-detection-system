FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./

RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache -e ".[all]" \
    || pip install --no-cache-dir -e "."

COPY src/ src/
COPY configs/ configs/
COPY monitoring/ monitoring/

# Expose the inference port
EXPOSE 8000

# Health check — Docker and orchestrators use this to know
# when the container is ready to receive traffic
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "65"]
