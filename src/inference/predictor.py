import pandas as pd
import numpy as np
import pickle
import logging
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import yaml
from pathlib import Path

from src.features.time_features import compute_time_features
from src.features.velocity_features import compute_velocity_features
from src.features.user_aggregations import compute_user_aggregations
from src.features.categorical_encoder import apply_target_encoder
from src.inference.schemas import TransactionInput, PredictionOutput, compute_risk_level

logger = logging.getLogger(__name__)


CONFIG_PATH = Path("configs/model_config.yaml")

MODEL_NAME = "fraud_detection_model"


class FraudPredictor:
    def __init__(self):
        self.model = None
        self.encoding = None
        self.feature_columns = None
        self.threshold = 0.5
        self.model_version = None
        self.run_id = None
        self.config = None
        self._loaded = False

    def load(self, model_uri: str = None) -> None:
        """
        Load all artifacts. Called once at application startup.
        """
        logger.info("Loading fraud predictor artifacts...")

        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                self.config = yaml.safe_load(f)

            tracking_uri = self.config.get(
                "mlflow_tracking_uri",
                "postgresql://postgres:admin@localhost:5432/mlflow_tracking",
            )

        else:
            tracking_uri = "postgresql://postgres:admin@localhost:5432/mlflow_tracking"
            self.config = {}

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
            logger.warning(
                "Loading from local path — version info unavailable. Using default threshold 0.5."
            )

        encodings_loaded = False
        features_loaded = False

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
                    line.strip() for line in f.readlines() if line.strip()
                ]

            logger.info(
                f"Loaded encodings for {len(self.encodings)} categorical columns."
            )
            logger.info(f"Loaded {len(self.feature_columns)} feature columns.")

            self._loaded = True
            logger.info(
                f"Predictor ready. Model version: {self.model_version}, Threshold: {self.threshold:.4f}"
            )

        except Exception as e:
            logger.info(
                f"Could not load from MLflow artifacts: {e}. Trying local disk."
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
            versions = client.get_latest_versions(
                model_name,
                stages=[stage_or_version] if not stage_or_version.isdigit() else None,
            )

            if stage_or_version.isdigit():
                # Specific version number requested
                version_obj = client.get_model_version(model_name, stage_or_version)
                self.model_version = stage_or_version
                run_id = version_obj.run_id
                self.run_id = run_id

            elif versions:
                version_obj = versions[0]
                self.model_version = version_obj.version
                run_id = version_obj.run_id
                self.run_id = run_id

            else:
                logger.warning(
                    f"No version found for stage {stage_or_version}. Using default threshold 0.5."
                )
                self.model_version = "unknown"
                return

            # Load threshold from the training run params
            run = client.get_run(run_id)
            threshold_str = run.data.params.get("optimal_threshold")
            if threshold_str is not None:
                self.threshold = float(threshold_str)
                logger.info(
                    f"Loaded threshold {self.threshold:.4f} from MLflow run {run_id[:8]}..."
                )
            else:
                logger.warning(
                    "optimal_threshold not found in run params. Using default 0.5."
                )
                self.threshold = 0.5

        except Exception as e:
            logger.warning(
                f"Could not fetch version/threshold from MLflow: {e}. Using defaults."
            )
            self.model_version = "unknown"
            self.threshold = 0.5

    def _transaction_to_dataframe(self, transaction: TransactionInput) -> pd.DataFrame:
        """
        Convert a single TransactionInput Pydantic object into a one-row pandas DataFrame.
        """

        data = transaction.model_dump()
        df = pd.DataFrame([data])
        df = df.where(df.notna(), other=np.nan)

        return df

    def _run_feature_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the feature engineering pipeline to raw transaction data.
        """
        logger.debug(f"Running feature pipeline on {len(df)} rows...")

        df = compute_time_features(df)
        df = compute_velocity_features(df)
        df = compute_user_aggregations(df)
        df = apply_target_encoder(df, self.encodings)

        return df

    def _select_model_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select exactly the columns the model was trained on, in the
        exact same order, handling missing and extra columns.
        """
        missing_cols = [c for c in self.feature_columns if c not in df.columns]

        if missing_cols:
            logger.debug(
                f"{len(missing_cols)} feature columns missing from input, filling with NaN: {missing_cols[:5]}..."
            )
            # Add all missing columns in one operation — avoids fragmentation
            missing_df = pd.DataFrame(
                np.nan,
                index=df.index,
                columns=missing_cols,
            )

            df = pd.concat([df, missing_df], axis=1)

        # Select in exact training order
        return df[self.feature_columns]

    def predict_single(self, transaction: TransactionInput) -> PredictionOutput:
        """
        Score a single transaction.
        """
        if not self._loaded:
            raise RuntimeError("Predictor is not loaded. Call predictor.load() first.")

        # Convert to DataFrame
        df = self._transaction_to_dataframe(transaction)
        print("After transaction_to_dataframe:", df.shape)

        # Run feature engineering
        df = self._run_feature_pipeline(df)
        print("After feature pipeline:", df.shape)

        # Select model features in correct order
        X = self._select_model_features(df)
        print("After select features:", X.shape)

        fraud_probability = float(self.model.predict_proba(X)[0, 1])

        # Apply threshold to get binary decision
        is_fraud = fraud_probability >= self.threshold

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

        # Build a multi-row DataFrame
        rows = [t.model_dump() for t in transactions]
        df = pd.DataFrame(rows).where(pd.DataFrame(rows).notna(), other=np.nan)

        # Run feature engineering on the full batch
        df = self._run_feature_pipeline(df)

        # Select model features
        X = self._select_model_features(df)

        # Batch predict — single model call
        fraud_probabilities = self.model.predict_proba(X)[:, 1]

        # Build output list preserving input order
        results = []
        for i, transaction in enumerate(transactions):
            prob = float(fraud_probabilities[i])
            results.append(
                PredictionOutput(
                    TransactionID=transaction.TransactionID,
                    fraud_probability=round(prob, 6),
                    is_fraud=prob >= self.threshold,
                    risk_level=compute_risk_level(prob),
                    threshold_used=round(self.threshold, 4),
                    model_version=str(self.model_version),
                )
            )

        flagged = sum(1 for r in results if r.is_fraud)
        logger.info(
            f"Batch scoring complete. Flagged {flagged}/{len(transactions)} as fraud ({flagged / len(transactions) * 100:.1f}%)."
        )

        return results


predictor = FraudPredictor()
