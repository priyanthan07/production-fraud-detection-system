import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.inference.predictor import predictor
from src.inference.schemas import (
    TransactionInput,
    BatchTransactionInput,
    PredictionOutput,
    BatchPredictionOutput,
    HealthResponse,
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
        logger.info("Server ready to accept requests.")
    except Exception as e:
        logger.error(
            f"Failed to load predictor at startup: {e}. "
            f"Server will start but /predict endpoints will fail."
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
    if not predictor._loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. The server may still be starting up. "
                "Try again in a few seconds."
            ),
        )
    
    start_time = time.time()
    
    try:
        result = predictor.predict_single(transaction)
    except Exception as e:
        logger.error(
            f"Prediction failed for TransactionID "
            f"{transaction.TransactionID}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )
    
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(f"TransactionID={transaction.TransactionID} fraud_probability={result.fraud_probability:.4f} is_fraud={result.is_fraud} latency={elapsed_ms:.1f}ms")
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
    """Score a batch of transactions in a single request."""
    
    if not predictor._loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )
 
    start_time = time.time()
    n = len(batch.transactions)
    
    try:
        predictions = predictor.predict_batch(batch.transactions)
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}",
        )
 
    flagged = sum(1 for p in predictions if p.is_fraud)
    elapsed_ms = (time.time() - start_time) * 1000
 
    logger.info(
        f"Batch scored {n} transactions in {elapsed_ms:.1f}ms. Flagged {flagged} as fraud ({flagged/n*100:.1f}%)."
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