import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Time windows in seconds
WINDOW_1HR = 3600
WINDOW_24HR = 86400
WINDOW_7DAY = 604800


def compute_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling velocity features for card1 and P_emaildomain.

    Velocity features capture how frequently a card or email is
    transacting within a time window. Fraudsters tend to make many
    transactions in a short period before the card is blocked.

    This function sorts by TransactionDT and uses a merge-asof approach
    to count transactions and sum amounts within each time window for
    each grouping key without using future data.

    """
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    logger.info("Computing card1 velocity features...")
    df = _velocity_for_key(df, group_key="card1")

    logger.info("Computing P_emaildomain velocity features...")
    df = _velocity_for_key(df, group_key="P_emaildomain")

    logger.info("Velocity features complete.")
    return df


def _velocity_for_key(df: pd.DataFrame, group_key: str) -> pd.DataFrame:
    """
    For each transaction compute rolling count and sum of TransactionAmt
    within 1hr, 24hr, and 7day windows grouped by group_key.

    Only uses past transactions to avoid data leakage.
    Transactions where group_key is NaN are skipped and get NaN features.
    """

    windows = {"1hr": WINDOW_1HR, "24hr": WINDOW_24HR, "7day": WINDOW_7DAY}

    # Prepare output columns
    for window_name in windows:
        df[f"{group_key}_count_{window_name}"] = np.nan
        df[f"{group_key}_amt_sum_{window_name}"] = np.nan

    # Skip rows where the grouping key is null
    valid_mask = df[group_key].notna()
    valid_df = df[valid_mask].copy()

    if len(valid_df) == 0:
        logger.warning(f"No valid rows for group_key={group_key}, skipping.")
        return df

    # Process each unique value of the group key
    unique_keys = valid_df[group_key].unique()

    results = []

    for key_value in unique_keys:
        key_df = valid_df[valid_df[group_key] == key_value].copy()
        key_df = key_df.sort_values("TransactionDT").reset_index(drop=True)

        n = len(key_df)
        times = key_df["TransactionDT"].values
        amounts = key_df["TransactionAmt"].values

        for window_name, window_seconds in windows.items():
            counts = np.zeros(n)
            amt_sums = np.zeros(n)

            for i in range(n):
                current_time = times[i]
                window_start = current_time - window_seconds

                # Only look at past transactions strictly before current
                past_mask = (times < current_time) & (times >= window_start)
                counts[i] = past_mask.sum()
                amt_sums[i] = amounts[past_mask].sum()

            key_df[f"{group_key}_count_{window_name}"] = counts
            key_df[f"{group_key}_amt_sum_{window_name}"] = amt_sums

        results.append(key_df)

    if results:
        result_df = pd.concat(results, ignore_index=True)

        # Merge back into original dataframe on TransactionID
        cols_to_merge = ["TransactionID"] + [
            col
            for col in result_df.columns
            if group_key in col and ("count" in col or "amt_sum" in col)
        ]

        df = df.drop(
            columns=[c for c in df.columns if group_key in c and ("count" in c or "amt_sum" in c)]
        )

        df = df.merge(result_df[cols_to_merge], on="TransactionID", how="left")

    return df
