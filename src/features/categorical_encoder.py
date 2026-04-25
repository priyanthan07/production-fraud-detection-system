import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

CATEGORICAL_COLUMNS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
]

def _compute_smoothed_means(
    df: pd.DataFrame,
    col: str,
    target_col: str,
    global_mean: float,
    smoothing: float,
)-> dict:
    """
        Compute smoothed category means from a given dataframe slice.
        Used both inside the CV loop (on train folds) and after the loop
        (on the full dataset for inference encodings).

        Extracted as a helper so the same formula is used in both places
        — no risk of the two diverging.
    """
    stats = (
        df.groupby(col)[target_col].agg(["mean", "count"]).reset_index()
    )
    
    stats.columns = [col, "cat_mean", "cat_count"]
    stats["smoothed_mean"] = (stats["cat_count"] * stats["cat_mean"] + smoothing * global_mean) / (stats["cat_count"] + smoothing)
    
    return stats.set_index(col)["smoothed_mean"].to_dict()


def fit_target_encoder(
    df: pd.DataFrame,
    categorical_cols: list,
    target_col: str = "isFraud",
    n_splits: int = 5,
    smoothing: float = 1.0,
) -> tuple:
    """
    Fit target encoding using cross-validation folds to prevent leakage.

    Two outputs:
    1. df — training dataframe with leak-free encoded columns produced
            by the CV loop. Each row's encoding was computed from the
            other folds only, never from the row itself.

    2. encodings — dict of category means computed from the FULL dataset.
            Used at inference time. At inference there is no label to
            leak, so using all rows gives the most stable estimate.
            These values will be close to but not identical to the
            CV values — that small gap is acceptable and expected.
    """
    df = df.copy()

    global_mean = df[target_col].mean()
    encodings = {}
    kf = KFold(n_splits=n_splits, shuffle=False)

    for col in categorical_cols:
        if col not in df.columns:
            logger.warning(f"Column {col} not found, skipping encoding.")
            continue

        logger.info(f"Fitting target encoding for {col}...")

        encoded = np.full(len(df), global_mean)

        for fold_num, (train_idx, val_idx) in enumerate(kf.split(df)):
            train_fold = df.iloc[train_idx]
            val_fold   = df.iloc[val_idx]

            # Smoothed category means from training fold ONLY
            fold_means = _compute_smoothed_means(
                train_fold, col, target_col, global_mean, smoothing
            )

            # Map validation rows using training-fold means
            val_mapped = (
                val_fold[col].map(fold_means).fillna(global_mean).values
            )

            encoded[val_idx] = val_mapped

            logger.debug(
                f"  Fold {fold_num + 1}: train={len(train_idx)} rows, val={len(val_idx)} rows, categories seen: {list(fold_means.keys())}"
            )

        # Store leak-free encoded column in df (used for model training)
        df[f"{col}_encoded"] = encoded

        inference_means = _compute_smoothed_means(
            df, col, target_col, global_mean, smoothing
        )

        encodings[col] = {
            "global_mean": global_mean,
            "category_means": inference_means,
        }

        logger.info(
            f"  '{col}' encoding complete. Categories: {len(inference_means)}, global_mean: {global_mean:.4f}"
        )
        
    return df, encodings


def apply_target_encoder(
    df: pd.DataFrame,
    encodings: dict,
) -> pd.DataFrame:
    """
    Apply pre-fitted target encodings to a dataframe.
    Used at inference time with encodings loaded from disk.
    Unseen categories fall back to the global mean.
    """
    df = df.copy()

    for col, encoding in encodings.items():
        if col not in df.columns:
            continue

        global_mean = encoding["global_mean"]
        category_means = encoding["category_means"]

        df[f"{col}_encoded"] = df[col].map(category_means).fillna(global_mean)

        logger.info(f"Applied target encoding for {col}")

    return df
