import pandas as pd
import logging
from src.features.velocity_features import compute_velocity_features
from src.features.time_features import compute_time_features
from src.features.user_aggregations import compute_user_aggregations
from src.features.categorical_encoder import (
    fit_target_encoder,
    apply_target_encoder,
    CATEGORICAL_COLUMNS,
)

logger = logging.getLogger(__name__)


def build_features(
    df: pd.DataFrame,
    encodings: dict = None,
    fit_encodings: bool = False,
) -> tuple:
    """
    Run the full feature engineering pipeline in the correct order.

    Order matters:
    1. Time features first because velocity and aggregations depend
    on TransactionDT being sorted and processed correctly
    2. Velocity features second
    3. User aggregations third
    4. Categorical encoding last because it needs the target column
    """
    logger.info("Starting feature engineering pipeline...")

    logger.info("Step 1: Time features")
    df = compute_time_features(df)

    logger.info("Step 2: Velocity features")
    df = compute_velocity_features(df)

    logger.info("Step 3: User aggregations")
    df = compute_user_aggregations(df)

    logger.info("Step 4: Categorical encoding")
    if fit_encodings:
        if "isFraud" not in df.columns:
            raise ValueError(
                "isFraud column required to fit target encodings. "
                "Cannot fit encodings on data without labels."
            )
        # fit_target_encoder now returns (df, encodings)
        df, encodings = fit_target_encoder(df, CATEGORICAL_COLUMNS)
        logger.info("Target encodings fitted.")

    if encodings is not None and not fit_encodings:
        # At inference time apply pre-fitted encodings
        df = apply_target_encoder(df, encodings)

    elif encodings is None and not fit_encodings:
        logger.warning(
            "No encodings provided and fit_encodings=False. "
            "Categorical columns will not be encoded."
        )

    logger.info("Feature engineering pipeline complete.")
    logger.info(f"Output dataframe shape: {df.shape}")

    return df, encodings
