import pandas as pd
import logging
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)


def build_catboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict = None,
) -> CatBoostClassifier:
    """
    Build and train a CatBoost model for fraud detection.

    CatBoost has excellent native handling of categorical features
    and is resistant to overfitting through its ordered boosting
    algorithm. auto_class_weights='Balanced' handles class imbalance
    by automatically computing class weights from the training data.
    """
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()

    logger.info(f"Class distribution: {n_negative} legitimate, {n_positive} fraud")

    default_params = {
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bylevel": 0.8,
        "min_data_in_leaf": 20,
        "auto_class_weights": "Balanced",
        "random_seed": 42,
        "eval_metric": "AUC",
        "early_stopping_rounds": 50,
        "verbose": 100,
    }

    if params is not None:
        default_params.update(params)

    logger.info("Building CatBoost model...")
    logger.info(f"Parameters: {default_params}")

    model = CatBoostClassifier(**default_params)

    return model


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict = None,
) -> CatBoostClassifier:
    """
    Train CatBoost with early stopping on validation set.
    """
    model = build_catboost_model(X_train, y_train, params)

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    logger.info("CatBoost training complete.")
    return model
