import logging
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Metric 1: Request counter
REQUEST_COUNT = Counter(
    name="fraud_requests_total",
    documentation="Total number of prediction requests received",
    labelnames=["endpoint", "status"],
)

# Metric 2: Request latency histogram
REQUEST_LATENCY = Histogram(
    name="fraud_request_latency_seconds",
    documentation="Request latency in seconds from receipt to response",
    labelnames=["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Metric 3: Fraud score distribution histogram
SCORE_DISTRIBUTION = Histogram(
    name="fraud_score_distribution",
    documentation=(
        "Distribution of fraud probability scores output by the model. "
        "Monitor for distribution shift as an early drift warning."
    ),
    labelnames=["endpoint"],
    buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)

# Metric 4: Fraud detection counter
FRAUD_DETECTIONS = Counter(
    name="fraud_detections_total",
    documentation="Total number of transactions flagged as fraud",
    labelnames=["endpoint", "risk_level"],
)

# Metric 5: Batch size histogram
BATCH_SIZE = Histogram(
    name="fraud_batch_size",
    documentation="Number of transactions per batch request",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000, 10000],
)

# Metric 6: Model version gauge
MODEL_VERSION = Gauge(
    name="fraud_model_version",
    documentation=(
        "Currently loaded model version number. "
        "Changes when a new model is promoted to Production."
    ),
)

# Metric 7: Error counter
ERROR_COUNT = Counter(
    name="fraud_errors_total",
    documentation="Total number of prediction errors",
    labelnames=["endpoint", "error_type"],
)

# Metric 8: Active requests gauge
ACTIVE_REQUESTS = Gauge(
    name="fraud_active_requests",
    documentation="Number of prediction requests currently being processed",
    labelnames=["endpoint"],
)

def record_prediction(
    endpoint: str,
    status_code: int,
    latency_seconds: float,
    fraud_probability: float,
    is_fraud: bool,
    risk_level: str,
) -> None:
    """ 
        Record all metrics for a single prediction request.
    """
    
    status_str = str(status_code)
    
    # Increment request counter
    REQUEST_COUNT.labels(endpoint=endpoint, status=status_str).inc()
    
    # Record latency
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency_seconds)
    
    # Record score in distribution histogram
    SCORE_DISTRIBUTION.labels(endpoint=endpoint).observe(fraud_probability)
 
    # If fraud was detected, increment fraud counter
    if is_fraud:
        FRAUD_DETECTIONS.labels(endpoint=endpoint, risk_level=risk_level).inc()
        
def record_batch_prediction(
    batch_size: int,
    endpoint: str,
    status_code: int,
    latency_seconds: float,
    fraud_probabilities: list,
    fraud_flags: list,
    risk_levels: list,
) -> None:
    """ 
        Record all metrics for a batch prediction request.
    """
    status_str = str(status_code)
    
    # Request-level metrics
    REQUEST_COUNT.labels(endpoint=endpoint, status=status_str).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency_seconds)
    BATCH_SIZE.observe(batch_size)
    
    for prob, flagged, risk in zip(fraud_probabilities, fraud_flags, risk_levels):
        SCORE_DISTRIBUTION.labels(endpoint=endpoint).observe(prob)
        if flagged:
            FRAUD_DETECTIONS.labels(endpoint=endpoint, risk_level=risk).inc()
            
def record_error(endpoint: str, error_type: str) -> None:
    """
        Record a prediction error.
    """
    ERROR_COUNT.labels(endpoint=endpoint, error_type=error_type).inc()
    logger.warning(f"Error recorded: endpoint={endpoint} type={error_type}")
    
def set_model_version(version: str) -> None:
    """ 
        Update the model version gauge.
    """
    try:
        MODEL_VERSION.set(float(version))
        logger.info(f"Model version gauge set to {version}")
    except (ValueError, TypeError):
        MODEL_VERSION.set(-1)
        logger.warning(
            f"Non-numeric model version '{version}'. Setting gauge to -1.")
        