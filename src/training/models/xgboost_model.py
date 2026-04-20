import pandas as pd
import logging
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def build_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    params: dict = None,
) -> XGBClassifier:
    """
    Build and train an XGBoost model for fraud detection.

    XGBoost handles missing values natively by learning the best
    direction to send NaN values at each split. We do not need
    to impute missing values before training.

    Class imbalance is handled via scale_pos_weight which tells
    the model to weight positive (fraud) examples more heavily.
    """

    # Compute scale_pos_weight from training data
    # This is the ratio of negative to positive examples
    # Tells XGBoost to weight fraud cases more heavily

    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale_pos_weight = n_negative / n_positive

    logger.info(f"Class distribution: {n_negative} legitimate, {n_positive} fraud")
    logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")

    default_params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "aucpr",
        "early_stopping_rounds": 50,
    }

    if params is not None:
        default_params.update(params)
        default_params["scale_pos_weight"] = scale_pos_weight

    logger.info("Training XGBoost model...")
    logger.info(f"Parameters: {default_params}")

    model = XGBClassifier(**default_params)
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict = None,
) -> XGBClassifier:
    """
    Train XGBoost with early stopping on validation set.

    Early stopping monitors AUC-PR on the validation set and
    stops training when it stops improving for 50 rounds.
    This prevents overfitting without manually tuning n_estimators.
    """

    model = build_xgboost_model(X_train, y_train, params)

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    best_iteration = model.best_iteration
    logger.info(f"XGBoost best iteration: {best_iteration}")

    return model
