"""
Warm Redis from historical training data.

Run ONCE after first training:
    python scripts/bootstrap_redis.py

Run again ONLY if Redis data is lost (e.g., docker compose down -v).
Do NOT run after retraining — Redis already contains live state
accumulated during production scoring.
"""

import logging
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    import yaml
    from src.feature_store.online_store import OnlineFeatureStore

    config_path = Path("configs/model_config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        redis_host = config.get("redis_host", "localhost")
        redis_port = int(config.get("redis_port", 6379))
    else:
        redis_host = "localhost"
        redis_port = 6379

    parquet_path = "data/processed/train_features.parquet"

    if not Path(parquet_path).exists():
        logger.error(
            f"{parquet_path} not found. "
            f"Run 'make train' first to generate processed features."
        )
        sys.exit(1)

    logger.info(f"Connecting to Redis at {redis_host}:{redis_port}...")
    store = OnlineFeatureStore(host=redis_host, port=redis_port)

    try:
        store.connect()
    except Exception as e:
        logger.error(
            f"Cannot connect to Redis: {e}. "
            f"Ensure Redis is running: docker compose up -d redis"
        )
        sys.exit(1)

    logger.info("Starting bootstrap from training data...")
    logger.info("This loads aggregated card/email statistics — not all 590K rows.")
    logger.info("Expected time: 5-15 minutes depending on dataset size.")

    store.bootstrap_from_parquet(parquet_path)

    logger.info("Bootstrap complete.")
    logger.info(
        "Redis now contains card and email history from training data. "
        "Inference server will serve correct velocity features."
    )


if __name__ == "__main__":
    main()