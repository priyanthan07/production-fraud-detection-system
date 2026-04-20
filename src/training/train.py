import pandas as pd
import logging
import pickle
import yaml
from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import warnings

# Suppress MLflow schema warnings about integer columns
warnings.filterwarnings(
    "ignore", message="Hint: Inferred schema contains integer column"
)

# Suppress MLflow pip requirements warning
logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)

# Suppress MLflow model metadata warning
logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)

from src.ingestion.loader import load_raw_data
from src.ingestion.validator import validate_raw_data
from src.features.pipeline import build_features
from src.training.models.xgboost_model import train_xgboost
from src.training.models.lightgbm_model import train_lightgbm
from src.training.models.catboost_model import train_catboost
from src.training.evaluator import evaluate_model
from src.training.threshold_optimizer import find_optimal_threshold
from src.explainability.shap_analysis import run_shap_analysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    with open("configs/model_config.yaml", "r") as f:
        return yaml.safe_load(f)


def get_feature_columns(df: pd.DataFrame, config: dict) -> list:
    """
    Return list of columns to use as features.

    Drops:
    - Columns explicitly listed in config drop_columns
    - Object dtype columns that were not encoded
    - V columns with null rate above null_threshold
    """
    drop_cols = set(config["drop_columns"])
    null_threshold = config["null_threshold"]

    feature_cols = []

    for col in df.columns:
        if col in drop_cols:
            continue

        # Drop object dtype columns that were not encoded
        if df[col].dtype == object:
            logger.warning(f"Dropping object column not in drop list: {col}")
            continue

        # Drop columns with too many nulls
        null_rate = df[col].isnull().mean()
        if null_rate > null_threshold:
            continue

        feature_cols.append(col)

    logger.info(f"Using {len(feature_cols)} feature columns")
    return feature_cols


def time_based_split(df: pd.DataFrame, train_ratio: float, target_col: str) -> tuple:
    """
    Split dataframe into train and validation using time order.

    Data is already sorted by TransactionDT from feature pipeline.
    We take first train_ratio rows for training and the rest for validation.

    This mirrors production deployment: train on past, predict on future.
    """
    split_idx = int(len(df) * train_ratio)

    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    logger.info(f"Train size: {len(train_df)} rows")
    logger.info(f"Validation size: {len(val_df)} rows")
    logger.info(f"Train fraud rate: {train_df[target_col].mean():.4f}")
    logger.info(f"Validation fraud rate: {val_df[target_col].mean():.4f}")

    return train_df, val_df


def load_or_compute_features(config: dict) -> tuple:
    """
    Load processed features from disk if they exist.
    Otherwise run the full feature pipeline and save to disk.

    Returns tuple of (feature dataframe, encodings dict)
    """
    processed_path = Path(config["processed_train_path"])
    encoding_path = Path(config["encodings_path"])

    processed_path.parent.mkdir(parents=True, exist_ok=True)

    if processed_path.exists() and encoding_path.exists():
        logger.info("Loading cached processed features from disk...")
        df = pd.read_parquet(processed_path)

        with open(encoding_path, "rb") as f:
            encodings = pickle.load(f)

        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df, encodings

    logger.info("No cached features found. Running feature pipeline...")
    logger.info("This will take several minutes on the full dataset.")

    raw_df = load_raw_data("data/raw")
    validate_raw_data(raw_df)

    df, encodings = build_features(raw_df, fit_encodings=True)

    with open(encoding_path, "wb") as f:
        pickle.dump(encodings, f)

    logger.info("Saving processed features to disk...")
    df.to_parquet(processed_path, index=False)

    logger.info(f"Saved to {processed_path}")
    logger.info(f"Saved encodings to {encoding_path}")

    return df, encodings


def train_and_evaluate_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict = None,
) -> tuple:
    """
    Train a single model, find optimal threshold, and evaluate.

    Returns tuple of (trained model, metrics dict, optimal threshold)
    """
    logger.info(f"Training {model_name}...")

    if model_name == "xgboost":
        model = train_xgboost(X_train, y_train, X_val, y_val, params)

    elif model_name == "lightgbm":
        model = train_lightgbm(X_train, y_train, X_val, y_val, params)

    elif model_name == "catboost":
        model = train_catboost(X_train, y_train, X_val, y_val, params)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    y_pred_proba = model.predict_proba(X_val)[:, 1]

    optimal_threshold = find_optimal_threshold(
        y_val.values,
        y_pred_proba,
        strategy="f1",
    )

    metrics = evaluate_model(
        y_val.values,
        y_pred_proba,
        threshold=optimal_threshold,
    )

    return model, metrics, optimal_threshold


def main():

    config = load_config()

    # Use local PostgreSQL for MLflow tracking and model registry
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])

    mlflow.set_experiment(config["mlflow_experiment_name"])

    # Load or compute features
    df, encodings = load_or_compute_features(config)

    # Get feature columns
    feature_cols = get_feature_columns(df, config)
    target_cols = config["target_col"]

    # Time based split
    train_df, val_df = time_based_split(df, config["train_ratio"], target_cols)

    X_train = train_df[feature_cols]
    y_train = train_df[target_cols]
    X_val = val_df[feature_cols]
    y_val = val_df[target_cols]

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_val shape: {X_val.shape}")

    # Train all three models
    model_names = ["xgboost", "lightgbm", "catboost"]

    # Log feature list as artifact
    feature_list_path = "data/processed/feature_columns.txt"

    with open(feature_list_path, "w") as f:
        f.write("\n".join(feature_cols))

    results = {}

    for model_name in model_names:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Training {model_name}")
        logger.info(f"{'=' * 50}")

        with mlflow.start_run(run_name=model_name):
            model, metrics, threshold = train_and_evaluate_model(
                model_name,
                X_train,
                y_train,
                X_val,
                y_val,
            )

            # Log parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("val_size", len(X_val))
            mlflow.log_param("n_features", len(feature_cols))
            mlflow.log_param("optimal_threshold", threshold)

            # Log metrics
            mlflow.log_metrics(metrics)

            X_val_float = X_val.astype(
                {col: "float64" for col in X_val.select_dtypes("integer").columns}
            )

            # Log model
            signature = infer_signature(X_val_float, model.predict_proba(X_val)[:, 1])

            mlflow.sklearn.log_model(
                model,
                artifact_path=model_name,
                signature=signature,
                registered_model_name=config["mlflow_model_name"],
            )

            # SHAP — runs inside the same MLflow run so plots and
            # the importance CSV are co-located with the model weights.
            logger.info(f"Running SHAP analysis for {model_name}...")
            try:
                shap_artifacts = run_shap_analysis(
                    model=model,
                    model_name=model_name,
                    X_val=X_val,
                    y_val=y_val,
                )
                logger.info(f"SHAP analysis complete for {model_name}.")
            except Exception as e:
                # SHAP failure must never abort the training run.
                # The model and metrics are already logged above.
                logger.warning(
                    f"SHAP analysis failed for {model_name}: {e}. "
                    f"Continuing without SHAP artifacts."
                )
                shap_artifacts = {}

            results[model_name] = {
                "model": model,
                "metrics": metrics,
                "threshold": threshold,
                "shap_artifacts": shap_artifacts,
            }

            logger.info(f"{model_name} results:")
            for metric_name, value in metrics.items():
                logger.info(f"  {metric_name}: {value}")

    # Find best model by AUC-PR
    best_model_name = max(results, key=lambda name: results[name]["metrics"]["auc_pr"])

    best_metrics = results[best_model_name]["metrics"]

    best_shap = results[best_model_name].get("shap_artifacts", {})
    importance_df = best_shap.get("importance_df")
    if importance_df is not None:
        logger.info(f"Top 10 features by SHAP ({best_model_name}):")
        logger.info(importance_df.head(10).to_string(index=False))

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"AUC-PR: {best_metrics['auc_pr']}")
    logger.info(f"AUC-ROC: {best_metrics['auc_roc']}")
    logger.info(f"F1: {best_metrics['f1']}")
    logger.info(f"Precision: {best_metrics['precision']}")
    logger.info(f"Recall: {best_metrics['recall']}")
    logger.info(f"{'=' * 50}\n")

    return results


if __name__ == "__main__":
    main()
