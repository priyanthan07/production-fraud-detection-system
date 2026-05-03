import logging

import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

logger = logging.getLogger(__name__)


def build_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict = None,
) -> LGBMClassifier:
    """
    Build and train a LightGBM model for fraud detection.

    LightGBM uses leaf-wise tree growth which makes it faster and
    often more accurate than level-wise growth used by XGBoost on
    large datasets. On 590,000 rows LightGBM is typically the
    fastest of the three models.

    is_unbalance=True automatically handles class imbalance by
    internally reweighting samples based on class frequencies.
    """
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()

    logger.info(f"Class distribution: {n_negative} legitimate, {n_positive} fraud")

    default_params = {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 5,
        "is_unbalance": True,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    if params is not None:
        default_params.update(params)

    logger.info("Building LightGBM model...")
    logger.info(f"Parameters: {default_params}")

    model = LGBMClassifier(**default_params)

    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict = None,
) -> LGBMClassifier:
    """
    Train LightGBM with early stopping on validation set.
    """
    model = build_lightgbm_model(X_train, y_train, params)

    callbacks = []

    try:
        callbacks = [
            early_stopping(stopping_rounds=100, verbose=True),
            log_evaluation(period=100),
        ]
    except ImportError:
        logger.warning("LightGBM callbacks not available. Training without early stopping.")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks if callbacks else None,
    )

    logger.info("LightGBM training complete.")
    return model
