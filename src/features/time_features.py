import pandas as pd
import logging

logger = logging.getLogger(__name__)

# TransactionDT starts at 86400 seconds (day 1)
DT_START = 86400

def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
        Compute time-based features from TransactionDT.

        TransactionDT is seconds elapsed since a reference point.
        Min value is 86400 (one day in seconds).
        We derive human-readable time features from this.
    """
    df = df.copy()
    
    logger.info("Computing time features...")
    
    # Convert to actual seconds from epoch-like reference
    # 86400 = seconds in a day
    df["hour_of_day"] = (df["TransactionDT"] // 3600) % 24
    df["day_of_week"] = (df["TransactionDT"] // 86400) % 7
    
    # Days since dataset start
    df["days_since_start"] = (df["TransactionDT"] - DT_START) / 86400
    
    # Time since last transaction per card1
    logger.info("Computing time since last transaction per card1...")
    df = df.sort_values("TransactionDT")
    df["time_since_last_txn_card1"] = (
        df.groupby("card1")["TransactionDT"]
        .diff()
        .fillna(-1)
    )
    
    # Days since card1 was first seen in the dataset
    logger.info("Computing days since card first seen...")
    card1_first_seen = (
        df.groupby("card1")["TransactionDT"]
        .transform("min")
    )
    
    df["days_since_card_first_seen"] = (
        (df["TransactionDT"] - card1_first_seen)/ 86400
    )
    
    # Is the transaction at night (between 11pm and 5am)
    df["is_night_transaction"] = (
        (df["hour_of_day"] >= 23) | (df["hour_of_day"] <= 5)
    ).astype(int)
    
    # Is the transaction on a weekend
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    logger.info("Time features complete.")
    return df
    