import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_raw_data(data_dir: str) -> pd.DataFrame:
    """
        Load and merge transaction and identity files.
        Not all transactions have identity records so we use a left join.
    """
    
    data_path = Path(data_dir)
    
    transaction_path = data_path / "train_transaction.csv"
    identity_path = data_path / "train_identity.csv"
    
    if not transaction_path.exists():
        raise FileNotFoundError(f"Transaction file not found: {transaction_path}")
    
    if not identity_path.exists():
        raise FileNotFoundError(f"Identity file not found : {identity_path}")
    
    logger.info("Loading transaction data...")
    transactions = pd.read_csv(transaction_path)
    logger.info(f"Transaction data loaded: {transactions.shape}")
    
    logger.info("Loading identity data...")
    identity = pd.read_csv(identity_path)
    logger.info(f"Identity data loaded: {identity.shape}")
    
    logger.info("Merging transaction and identity data...")
    merged = transactions.merge(identity, on="TransactionID", how="left")
    logger.info(f"Merged data shape: {merged.shape}")
    
    # Sanity check: merged row count must equal transaction row count
    assert len(merged) == len(transactions), (
        f"Merge produced unexpected row count. "
        f"Expected {len(transactions)}, got {len(merged)}"
    )
    
    logger.info("Data loading complete.")
    return merged

def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return all feature columns, excluding identifiers and target.
    """
    exclude = ["TransactionID", "isFraud", "TransactionDT"]
    return [col for col in df.columns if col not in exclude]

def get_target_column() -> str:
    return "isFraud"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_raw_data("data/raw")
    print(df.head())
    print(df["isFraud"].value_counts(normalize=True))