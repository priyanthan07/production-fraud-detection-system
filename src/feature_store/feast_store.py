import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_REPO_PATH = str(Path(__file__).resolve().parents[2] / "feature_repo")


class FeastFeatureStore:
    """
    Wrapper around the Feast FeatureStore for fraud detection.

    Uses Feast for:
    - Feature registry management
    - Point-in-time correct training data retrieval
    - Batch materialization of static features to Redis
    - Online lookup of static features at inference time
    """

    def __init__(self, repo_path: str = FEATURE_REPO_PATH):
        self.repo_path = repo_path
        self._store = None

    def connect(self) -> None:
        """
        Initialize the Feast FeatureStore.
        Reads feature_store.yaml from repo_path.
        """
        from feast import FeatureStore

        self._store = FeatureStore(repo_path=self.repo_path)
        logger.info(f"Feast FeatureStore initialized from {self.repo_path}")

    @property
    def store(self):
        if self._store is None:
            raise RuntimeError("FeastFeatureStore not connected. Call connect() first.")
        return self._store

    # Training: point-in-time correct feature retrieval
    def get_training_features(
        self,
        entity_df: pd.DataFrame,
        feature_refs: List[str],
    ) -> pd.DataFrame:
        """
        Retrieve historical features with point-in-time correctness.

        Feast ensures that for each row in entity_df, it only uses
        feature values that were available at that row's timestamp.
        This prevents data leakage in training data.
        """
        logger.info(f"Fetching historical features for {len(entity_df)} entity rows...")

        training_df = self.store.get_historical_features(
            entity_df=entity_df,
            features=feature_refs,
        ).to_df()

        logger.info(f"Retrieved {len(training_df)} rows with {len(training_df.columns)} columns.")

        return training_df

    # Materialization: push batch features from parquet to Redis
    def materialize(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> None:
        """
        Materialize feature values from the offline parquet store into the Redis online store.

        This is the Feast equivalent of "sync offline to online".
        Run this:
        - After first training (to populate Redis with initial values)
        - After retraining (to refresh Redis with updated features)

        Feast handles point-in-time correctness during materialization:
        it takes the latest value for each entity key from the parquet.

        Note: This does NOT populate velocity/aggregation features.
        Those are handled by online_store.py bootstrap_from_parquet().
        """

        if end_date is None:
            end_date = datetime.utcnow()

        logger.info(f"Materializing features to Redis online store up to {end_date.isoformat()}...")

        if start_date:
            self.store.materialize(start_date=start_date, end_date=end_date)
        else:
            self.store.materialize_incremental(end_date=end_date)

        logger.info("Materialization complete.")

    # Online serving: fetch static features at inference time
    def get_online_features(
        self,
        card_id: Optional[int],
        email_domain: Optional[str],
    ) -> dict:
        """
        Fetch static features from the Redis online store for a single entity.

        These are the pre-computed features that do not change per-transaction:
        categorical encodings, and the latest known C/D features per card.

        Note: Velocity and aggregation features (card1_count_1hr,
        card1_amt_zscore etc.) are NOT fetched here — they come from
        online_store.py which uses Redis sorted sets for real-time computation.

        Returns:
            dict of feature_name → value for all registered static features.
            Missing values are returned as None — predictor fills with NaN.
        """
        entity_rows = []
        feature_refs = []

        if card_id is not None:
            entity_rows.append({"card1": card_id})
            feature_refs.extend(
                [
                    "card_transaction_features:C1",
                    "card_transaction_features:C2",
                    "card_transaction_features:C3",
                    "card_transaction_features:C4",
                    "card_transaction_features:C5",
                    "card_transaction_features:C6",
                    "card_transaction_features:C7",
                    "card_transaction_features:C8",
                    "card_transaction_features:C9",
                    "card_transaction_features:C10",
                    "card_transaction_features:C11",
                    "card_transaction_features:C12",
                    "card_transaction_features:C13",
                    "card_transaction_features:C14",
                    "card_transaction_features:D1",
                    "card_transaction_features:D2",
                    "card_transaction_features:D3",
                    "card_transaction_features:D4",
                    "card_transaction_features:D5",
                    "card_transaction_features:D10",
                    "card_transaction_features:D11",
                    "card_transaction_features:D15",
                    "card_transaction_features:ProductCD_encoded",
                    "card_transaction_features:card4_encoded",
                    "card_transaction_features:card6_encoded",
                    "card_transaction_features:M1_encoded",
                    "card_transaction_features:M2_encoded",
                    "card_transaction_features:M3_encoded",
                    "card_transaction_features:M4_encoded",
                    "card_transaction_features:M5_encoded",
                    "card_transaction_features:M6_encoded",
                ]
            )

        if email_domain is not None:
            entity_rows.append({"P_emaildomain": email_domain})
            feature_refs.extend(
                [
                    "email_transaction_features:P_emaildomain_encoded",
                    "email_transaction_features:R_emaildomain_encoded",
                ]
            )

        if not entity_rows:
            return {}

        try:
            response = self.store.get_online_features(
                features=feature_refs,
                entity_rows=entity_rows,
            ).to_dict()

            # Flatten list values to scalars (Feast returns lists per entity)
            flat = {}
            for key, values in response.items():
                if key in ("card1", "P_emaildomain"):
                    continue
                # Take first non-None value from the list
                for v in values:
                    if v is not None:
                        flat[key] = v
                        break
                else:
                    flat[key] = None

            return flat

        except Exception as e:
            logger.error(f"Feast online feature lookup failed: {e}")
            return {}

    def is_healthy(self) -> bool:
        """Check if Feast store is initialized."""
        return self._store is not None


# Module-level singleton
feast_store = FeastFeatureStore()
