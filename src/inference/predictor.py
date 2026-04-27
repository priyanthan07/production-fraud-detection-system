import pandas as pd
import numpy as np
import pickle
import logging
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import yaml
from pathlib import Path
from typing import Optional

from src.feature_store.feast_store import feast_store
from src.feature_store.online_store import feature_store
from src.features.categorical_encoder import apply_target_encoder
from src.features.time_features import compute_time_features
from src.inference.schemas import TransactionInput, PredictionOutput, compute_risk_level

logger = logging.getLogger(__name__)


CONFIG_PATH = Path("configs/model_config.yaml")
MODEL_NAME = "fraud_detection_model"


class FraudPredictor:
    def __init__(self):
        self.model = None
        self.encoding = None
        self.feature_columns: list = []
        self.threshold = 0.5
        self.model_version = None
        self.run_id = None
        self.config = None
        self._loaded = False
        self._redis_ok        = False
        self._feast_ok        = False

    def load(self, model_uri: str = None) -> None:
        """
        Load all artifacts. Called once at application startup.
        """
        logger.info("Loading fraud predictor artifacts...")

        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                self.config = yaml.safe_load(f)
        
        else:
            self.config = {}

        tracking_uri = self.config.get(
            "mlflow_tracking_uri",
            "postgresql://postgres:admin@localhost:5432/mlflow_tracking",
        )

        mlflow.set_tracking_uri(tracking_uri)

        # Load model from MLflow
        if model_uri is None:
            model_uri = f"models:/{MODEL_NAME}/Production"

        logger.info(f"Loading model from: {model_uri}")
        self.model = mlflow.sklearn.load_model(model_uri)
        logger.info("Model loaded successfully.")

        if model_uri.startswith("models:/"):
            self._load_version_and_threshold(model_uri)
        else:
            self.model_version = "local"

        if not self.run_id:
            raise RuntimeError(
                "No run_id available. Cannot download artifacts. "
                "Make sure a model is registered in MLflow."
            )

        try:
            client = MlflowClient()
            artifact_dir = client.download_artifacts(
                self.run_id, "features", dst_path="/tmp/fraud_artifacts"
            )
            artifact_path = Path(artifact_dir)

            with open(artifact_path / "encodings.pkl", "rb") as f:
                self.encodings = pickle.load(f)

            with open(artifact_path / "feature_columns.txt") as f:
                self.feature_columns = [
                    line.strip() for line in f if line.strip()
                ]

            logger.info(f"Loaded encodings for {len(self.encodings)} categorical columns.")
            logger.info(f"Loaded {len(self.feature_columns)} feature columns.")

        except Exception as e:
            logger.info(f"Could not load from MLflow artifacts: {e}.")
            
        # Connect to Feast feature store (static features via Redis)
        try:
            feast_store.connect()
            self._feast_ok = True
            logger.info("Feast feature store connected.")
            
        except Exception as e:
            self._feast_ok = False
            logger.error(
                f"Feast connection failed: {e}. "
                f"Static features will be missing from Redis. "
                f"Run: python scripts/feast_apply.py && python scripts/feast_materialize.py"
            )
            
        # Connect to custom Redis feature store (velocity + aggregations)
        redis_host = self.config.get("redis_host", "localhost")
        redis_port = int(self.config.get("redis_port", 6379))
        feature_store.host = redis_host
        feature_store.port = redis_port

        try:
            feature_store.connect()
            self._redis_ok = True
            logger.info("Custom Redis feature store (velocity/aggregations) connected.")
        except Exception as e:
            self._redis_ok = False
            logger.error(
                f"Custom Redis connection failed: {e}. "
                f"Velocity and aggregation features will be 0/NaN. "
                f"Run: python scripts/bootstrap_redis.py"
            )

        self._loaded = True
        logger.info(
            f"FraudPredictor ready. "
            f"Model: {self.model_version}, "
            f"Threshold: {self.threshold:.4f}, "
            f"Feast: {'OK' if self._feast_ok else 'UNAVAILABLE'}, "
            f"Redis: {'OK' if self._redis_ok else 'UNAVAILABLE'}"
        )
        

    def _load_version_and_threshold(self, model_uri: str) -> str:
        """
        Fetch the model version number and optimal threshold from
        the MLflow registry. Called during load() when using a
        registry URI.
        """

        client = MlflowClient()

        parts = model_uri.replace("models:/", "").split("/")
        model_name = parts[0]
        stage_or_version = parts[1] if len(parts) > 1 else "Production"

        try:
            if stage_or_version.isdigit():
                version_obj = client.get_model_version(model_name, stage_or_version)
            
            else:
                versions = client.get_latest_versions(
                    model_name,
                    stages=[stage_or_version]
                )
                
                if not versions:
                    logger.warning(f"No version for stage '{stage_or_version}'.")
                    self.model_version = "unknown"
                    self.threshold     = 0.5
                    return
                version_obj = versions[0]

            self.model_version = version_obj.version
            self.run_id        = version_obj.run_id

            run = client.get_run(self.run_id)
            threshold_str = run.data.params.get("optimal_threshold")
            self.threshold = float(threshold_str) if threshold_str else 0.5
            logger.info(f"Threshold {self.threshold:.4f} from run {self.run_id[:8]}...")

        except Exception as e:
            logger.warning(
                f"Could not fetch version/threshold from MLflow: {e}. Using defaults."
            )
            self.model_version = "unknown"
            self.threshold = 0.5
            
    def _run_stateless_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Compute features that require only the current transaction row.

            Computed locally — no Redis, no Feast:
            - hour_of_day, day_of_week, days_since_start
            - is_night_transaction, is_weekend
            - ProductCD_encoded, card6_encoded, etc. (from encodings.pkl)

            NOT computed here:
            - Velocity features → custom Redis (online_store.py)
            - Aggregation features → custom Redis (online_store.py)
            - time_since_last_txn_card1 → custom Redis
            - days_since_card_first_seen → custom Redis
        """
        df = compute_time_features(df)
        df = apply_target_encoder(df, self.encodings)
        return df

    def _select_model_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Select exactly the columns the model was trained on,
            in exact training order. Fill missing with NaN.
        """
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            logger.debug(
                f"{len(missing)} feature columns missing, filling with NaN. "
                f"First 5: {missing[:5]}"
            )
            missing_df = pd.DataFrame(np.nan, index=df.index, columns=missing)
            df = pd.concat([df, missing_df], axis=1)
        return df[self.feature_columns]

    def predict_single(self, transaction: TransactionInput) -> PredictionOutput:
        """
        Score a single transaction.
        """
        if not self._loaded:
            raise RuntimeError("Predictor is not loaded. Call predictor.load() first.")

        current_time = float(transaction.TransactionDT)
        current_amt  = float(transaction.TransactionAmt)
        
        # Convert to DataFrame
        raw_dict = transaction.model_dump()
        df = pd.DataFrame([raw_dict])
        df = df.where(df.notna(), other=np.nan)
        df = self._run_stateless_features(df)
        print("After transaction_to_dataframe:", df.shape)

        # Step 2: Fetch velocity + aggregation features from custom Redis
        card_features  = feature_store.get_card_features(
            card_id      = transaction.card1,
            current_time = current_time,
            current_amt  = current_amt,
        )
        email_features = feature_store.get_email_features(
            email_domain = transaction.P_emaildomain,
            current_time = current_time,
            current_amt  = current_amt,
        )
        all_online_features = {**card_features, **email_features}

        # Step 3: Inject velocity/aggregation features into dataframe
        for feat_name, feat_val in all_online_features.items():
            df[feat_name] = feat_val
        
        # Select model features in correct order
        X = self._select_model_features(df)
        print("After select features:", X.shape)

        fraud_probability = float(self.model.predict_proba(X)[0, 1])

        # Apply threshold to get binary decision
        is_fraud = fraud_probability >= self.threshold

        # Step 5: Update custom Redis AFTER scoring (critical)
        feature_store.update(
            transaction_id   = transaction.TransactionID,
            card_id          = transaction.card1,
            email_domain     = transaction.P_emaildomain,
            amount           = current_amt,
            transaction_time = current_time,
        )

        # Step 6: Log scored features to disk for future retraining
        feature_store.log_scored_features(
            transaction_id    = transaction.TransactionID,
            card_id           = transaction.card1,
            email_domain      = transaction.P_emaildomain,
            amount            = current_amt,
            transaction_time  = current_time,
            features          = all_online_features,
            fraud_probability = fraud_probability,
            is_fraud          = is_fraud,
            model_version     = str(self.model_version),
        )

        # Step 7: Log raw input for audit
        raw_dict["fraud_probability"] = round(fraud_probability, 6)
        raw_dict["is_fraud"]          = is_fraud
        raw_dict["model_version"]     = str(self.model_version)
        feature_store.log_raw_transaction(raw_dict)

        # Step 8: Log for drift detection
        feature_store.log_prediction_for_drift(
            transaction_id   = transaction.TransactionID,
            transaction_time = current_time,
            features         = all_online_features,
        )

        logger.debug(
            f"TransactionID={transaction.TransactionID} "
            f"prob={fraud_probability:.4f} fraud={is_fraud}"
        )
        
        return PredictionOutput(
            TransactionID=transaction.TransactionID,
            fraud_probability=round(fraud_probability, 6),
            is_fraud=is_fraud,
            risk_level=compute_risk_level(fraud_probability),
            threshold_used=round(self.threshold, 4),
            model_version=str(self.model_version),
        )

    def predict_batch(
        self,
        transactions: list,
    ) -> list:
        """
        Score a batch of transactions efficiently.
        """

        if not self._loaded:
            raise RuntimeError("Predictor is not loaded. Call predictor.load() first.")

        if not transactions:
            return []

        logger.info(f"Scoring batch of {len(transactions)} transactions...")

        # Sort chronologically for correct velocity within batch
        sorted_txns = sorted(transactions, key=lambda t: t.TransactionDT)

        results_map: dict = {}
        for txn in sorted_txns:
            result = self.predict_single(txn)
            results_map[txn.TransactionID] = result

        # Return in original input order
        results = [results_map[t.TransactionID] for t in transactions]

        flagged = sum(1 for r in results if r.is_fraud)
        logger.info(
            f"Batch complete. {flagged}/{len(transactions)} flagged."
        )
        return results


predictor = FraudPredictor()
