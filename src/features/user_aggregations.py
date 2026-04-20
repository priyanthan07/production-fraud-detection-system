import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_user_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-user historical aggregations and deviation features.

    For each transaction we compute:
    - Historical mean and std of TransactionAmt per card1
    - Deviation of current transaction from that baseline
    - Transaction count per card1 up to current point

    These features encode individual user context so the global model
    can reason about whether a transaction is unusual for that specific user.

    All aggregations are computed using only past data to prevent leakage.
    """

    df = df.sort_values("TransactionDT").reset_index(drop=True)
    df = df.copy()

    # Expanding mean and std per card1 using only past transactions
    # shift(1) ensures we only use data before the current transaction
    df["card1_amt_mean"] = df.groupby("card1")["TransactionAmt"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    df["card1_amt_std"] = df.groupby("card1")["TransactionAmt"].transform(
        lambda x: x.shift(1).expanding().std()
    )

    # Transaction count per card1 up to but not including current transaction
    df["card1_txn_count"] = df.groupby("card1")["TransactionAmt"].transform(
        lambda x: x.shift(1).expanding().count()
    )

    # Deviation of current amount from historical mean
    # This is the key signal: how unusual is this transaction for this card?
    df["card1_amt_deviation"] = df["TransactionAmt"] - df["card1_amt_mean"]

    # Normalized deviation (z-score)
    # Avoid division by zero for cards with only one transaction
    df["card1_amt_zscore"] = np.where(
        df["card1_amt_std"] > 0, df["card1_amt_deviation"] / df["card1_amt_std"], 0.0
    )

    # Same aggregations for P_emaildomain
    logger.info("Computing email domain aggregation features...")

    valid_email_mask = df["P_emaildomain"].notna()

    df["email_amt_mean"] = np.nan
    df["email_amt_std"] = np.nan

    df.loc[valid_email_mask, "email_amt_mean"] = (
        df[valid_email_mask]
        .groupby("P_emaildomain")["TransactionAmt"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    df.loc[valid_email_mask, "email_amt_std"] = (
        df[valid_email_mask]
        .groupby("P_emaildomain")["TransactionAmt"]
        .transform(lambda x: x.shift(1).expanding().std())
    )

    df["email_amt_deviation"] = df["TransactionAmt"] - df["email_amt_mean"]

    df["email_amt_zscore"] = np.where(
        df["email_amt_std"] > 0, df["email_amt_deviation"] / df["email_amt_std"], 0.0
    )

    logger.info("User aggregation features complete.")
    return df
