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
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]

def fit_target_encoder(
    df: pd.DataFrame,
    categorical_cols: list,
    target_col: str = "isFraud",
    n_splits: int = 5,
    smoothing: float = 1.0,
) -> tuple:
    """
        Fit target encoding using cross-validation folds to prevent leakage.

        For each categorical column, compute the mean target value per category.
        Cross-validation ensures the encoding for a row is computed from
        other folds, never from the row itself.

        Smoothing blends the category mean with the global mean to handle
        rare categories with very few samples.

        Returns a dictionary of encodings that can be saved and reused at
        inference time.
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
        
        for train_idx, val_idx in kf.split(df):
            train_fold = df.iloc[train_idx]
            val_fold = df.iloc[val_idx]
            
            # Compute category means on training fold only
            category_stats = (
                train_fold.groupby(col)[target_col]
                .agg(["mean", "count"])
                .reset_index()
            )

            category_stats.columns = [col, "cat_mean", "cat_count"]
            
            # Apply smoothing: blend category mean with global mean
            # Categories with few samples pull toward global mean
            
            category_stats["smoothed_mean"] = (category_stats["cat_count"] * category_stats["cat_mean"] + smoothing * global_mean) / (category_stats["cat_count"] + smoothing)

            # Map encoded values to validation fold
            val_mapped = val_fold[[col]].merge(
                category_stats[[col, "smoothed_mean"]],
                on=col,
                how="left"
            )["smoothed_mean"].fillna(global_mean).values
            
            encoded[val_idx] = val_mapped
        
        df[f"{col}_encoded"] = encoded
            
        encodings[col] = {
            "global_mean" : global_mean,
            "category_means" : (
                df.groupby(col)[target_col]
                .agg(["mean", "count"])
                .reset_index()
                .assign(
                    smoothed_mean = lambda x : (
                        (x["count"] * x["mean"] + smoothing *global_mean) / (x["count"] + smoothing)
                    )
                )
                .set_index(col)["smoothed_mean"]
                .to_dict()
            )
        }
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
        
        df[f"{col}_encoded"] = (
            df[col].map(category_means).fillna(global_mean)
        )
        
        logger.info(f"Applied target encoding for {col}")

    return df
    