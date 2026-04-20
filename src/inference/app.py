import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.inference.predictor import predictor
from src.inference.schemas import (
    TransactionInput,
    BatchTransactionInput,
    PredictionOutput,
    BatchPredictionOutput,
    HealthResponse,
)

from src.monitoring.metrics_exporter import (
    record_prediction,
    record_batch_prediction,
    record_error,
    set_model_version,
    ACTIVE_REQUESTS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Lifespan — startup and shutdown logic
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs on server startup and shutdown."""

    logger.info("Starting fraud detection inference server...")
    try:
        predictor.load()
        set_model_version(str(predictor.model_version))
        logger.info("Server ready to accept requests.")
    except Exception as e:
        logger.error(
            f"Failed to load predictor at startup: {e}. Server will start but /predict endpoints will fail."
        )
    yield
    # Shutdown logic would go here if needed
    logger.info("Inference server shutting down.")


app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time fraud scoring for financial transactions. Returns a fraud probability and binary decision for each transaction."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"],
    summary="Check if the server and model are ready",
)
async def health_check():
    return HealthResponse(
        status="healthy" if predictor._loaded else "model_not_loaded",
        model_loaded=predictor._loaded,
        model_version=str(predictor.model_version or "none"),
        threshold=predictor.threshold,
    )


@app.get(
    "/metrics",
    tags=["Monitoring"],
    summary="Prometheus metrics",
    include_in_schema=True,
)
async def metrics():
    """
    Expose all Prometheus metrics in text format.
    Scraped by Prometheus every 15 seconds as configured in prometheus.yml.
    Do not expose this endpoint publicly in production.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post(
    "/predict",
    response_model=PredictionOutput,
    tags=["Scoring"],
    summary="Score a single transaction for fraud",
    responses={
        200: {"description": "Fraud score computed successfully"},
        422: {"description": "Invalid input — see detail for field errors"},
        503: {"description": "Model not loaded"},
    },
)
async def predict_single(transaction: TransactionInput):
    """
    Score a single transaction and return a fraud probability.
    """

    endpoint = "/predict"

    if not predictor._loaded:
        record_error(endpoint, "model_not_loaded")
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. The server may still be starting up. "
                "Try again in a few seconds."
            ),
        )

    ACTIVE_REQUESTS.labels(endpoint=endpoint).inc()

    start_time = time.time()

    try:
        result = predictor.predict_single(transaction)
    except Exception as e:
        latency = time.time() - start_time
        ACTIVE_REQUESTS.labels(endpoint=endpoint).dec()
        record_error(endpoint, "prediction_error")

        record_prediction(
            endpoint=endpoint,
            status_code=500,
            latency_seconds=latency,
            fraud_probability=0.0,
            is_fraud=False,
            risk_level="LOW",
        )

        logger.error(
            f"Prediction failed for TransactionID {transaction.TransactionID}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )

    latency = time.time() - start_time
    ACTIVE_REQUESTS.labels(endpoint=endpoint).dec()

    record_prediction(
        endpoint=endpoint,
        status_code=200,
        latency_seconds=latency,
        fraud_probability=result.fraud_probability,
        is_fraud=result.is_fraud,
        risk_level=result.risk_level,
    )

    logger.info(
        f"TransactionID={transaction.TransactionID} fraud_probability={result.fraud_probability:.4f} "
        f"is_fraud={result.is_fraud} latency={latency * 1000:.1f}ms"
    )

    return result


@app.post(
    "/predict/batch",
    response_model=BatchPredictionOutput,
    tags=["Scoring"],
    summary="Score a batch of transactions for fraud",
    responses={
        200: {"description": "All transactions scored successfully"},
        422: {"description": "Invalid input — see detail for field errors"},
        503: {"description": "Model not loaded"},
    },
)
async def predict_batch(batch: BatchTransactionInput):
    """
    Score a batch of transactions in a single request.
    """

    endpoint = "/predict/batch"

    if not predictor._loaded:
        record_error(endpoint, "model_not_loaded")
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    ACTIVE_REQUESTS.labels(endpoint=endpoint).inc()

    start_time = time.time()
    n = len(batch.transactions)

    try:
        predictions = predictor.predict_batch(batch.transactions)
    except Exception as e:
        latency = time.time() - start_time
        ACTIVE_REQUESTS.labels(endpoint=endpoint).dec()
        record_error(endpoint, "prediction_error")

        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}",
        )

    flagged = sum(1 for p in predictions if p.is_fraud)

    latency = time.time() - start_time
    ACTIVE_REQUESTS.labels(endpoint=endpoint).dec()

    record_batch_prediction(
        batch_size=n,
        endpoint=endpoint,
        status_code=200,
        latency_seconds=latency,
        fraud_probabilities=[p.fraud_probability for p in predictions],
        fraud_flags=[p.is_fraud for p in predictions],
        risk_levels=[p.risk_level for p in predictions],
    )

    logger.info(
        f"Batch scored {n} transactions in {latency * 1000:.1f}ms. Flagged {flagged} as fraud ({flagged / n * 100:.1f}%)."
    )

    return BatchPredictionOutput(
        predictions=predictions,
        total_transactions=n,
        flagged_as_fraud=flagged,
        fraud_rate_in_batch=round(flagged / n, 4),
        model_version=str(predictor.model_version),
    )


if __name__ == "__main__":
    uvicorn.run(
        "src.inference.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
