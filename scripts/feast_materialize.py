"""
    Materialize features from offline parquet store into Redis online store.

    Run after feast_apply.py and after training data exists:
        python scripts/feast_materialize.py

    This pushes the latest feature values from data/processed/train_features.parquet
    into Redis so the inference server can look them up instantly.

    Run again after each retraining to refresh Redis with updated feature values.
    Unlike bootstrap_redis.py (which handles velocity/aggregation custom logic),
    this handles Feast-managed static features.
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.feature_store.feast_store import FeastFeatureStore
    
    parquet_path = Path("data/processed/train_features.parquet")
    
    if not parquet_path.exists():
        logger.error(
            f"{parquet_path} not found. Run 'make train' first to generate processed features."
        )
        sys.exit(1)
        
    logger.info("Initializing Feast feature store...")
    store = FeastFeatureStore()
    store.connect()

    logger.info(
        "Materializing features from offline store to Redis online store..."
    )
    logger.info(
        "This pushes static features (C columns, D columns, encoded categoricals) "
        "into Redis. Velocity and aggregation features are handled separately by "
        "scripts/bootstrap_redis.py."
    )

    try:
        store.materialize()
        logger.info("Feast materialization complete.")
        logger.info(
            "Static features are now available in Redis for inference. "
            "Run 'python scripts/bootstrap_redis.py' to also populate "
            "velocity and aggregation features."
        )
    except Exception as e:
        logger.error(f"Materialization failed: {e}")
        logger.error(
            "Ensure Redis is running: docker compose up -d redis"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
        