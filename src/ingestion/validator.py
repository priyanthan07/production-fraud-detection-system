import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

def validate_raw_data(df: pd.DataFrame, report_dir: str = "data/validation_reports") -> bool:
    """
        Run data validation checks on the merged dataset.
        Returns True if all checks pass.
        Raises ValueError if any critical check fails.
    """
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    
    errors = []
    warnings = []
    
    # ----------------------------------------------------------------
    # Check 1: Required columns exist
    # ----------------------------------------------------------------
    required_columns = [
        "TransactionID",
        "isFraud",
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    else:
        logger.info("Check 1 passed: All required columns present")
        
    # ----------------------------------------------------------------
    # Check 2: Dataset is not empty
    # ----------------------------------------------------------------
    if len(df) == 0:
        errors.append("Dataset is empty")
    else:
        logger.info(f"Check 2 passed: Dataset has {len(df)} rows")
        
    # ----------------------------------------------------------------
    # Check 3: No duplicate TransactionIDs
    # ----------------------------------------------------------------
    duplicate_count = df["TransactionID"].duplicated().sum()
    if duplicate_count > 0:
        errors.append(f"Found {duplicate_count} duplicate TransactionIDs")
    else:
        logger.info("Check 3 passed: No duplicate TransactionIDs")
        
    # ----------------------------------------------------------------
    # Check 4: Target column isFraud only contains 0 and 1
    # ----------------------------------------------------------------
    if "isFraud" in df.columns:
        invalid_target = df[~df["isFraud"].isin([0, 1])]
        if len(invalid_target) > 0:
            errors.append(f"isFraud column contains values other than 0 and 1")
        else:
            logger.info("Check 4 passed: isFraud column values are valid")
    
    # ----------------------------------------------------------------
    # Check 5: Fraud rate is within expected range (1% to 10%)
    # ----------------------------------------------------------------
    if "isFraud" in df.columns:
        fraud_rate = df["isFraud"].mean()
        if fraud_rate < 0.01 or fraud_rate > 0.10:
            warnings.append(f"Fraud rate {fraud_rate:.4f} is outside expected range 0.01 to 0.10")
        else:
            logger.info(f"Check 5 passed: Fraud rate is {fraud_rate:.4f}")
            
    # ----------------------------------------------------------------
    # Check 6: TransactionAmt is non-negative
    # ----------------------------------------------------------------
    if "TransactionAmt" in df.columns:
        negative_amounts = (df["TransactionAmt"] < 0).sum()
        if negative_amounts > 0:
            errors.append(f"Found {negative_amounts} negative TransactionAmt values")
        else:
            logger.info("Check 6 passed: All TransactionAmt values are non-negative")
            
    # ----------------------------------------------------------------
    # Check 7: TransactionAmt null rate is below 5%
    # ----------------------------------------------------------------
    if "TransactionAmt" in df.columns:
        null_rate = df["TransactionAmt"].isnull().mean()
        if null_rate > 0.05:
            errors.append(
                f"TransactionAmt null rate {null_rate:.4f} exceeds 5% threshold"
            )
        else:
            logger.info(f"Check 7 passed: TransactionAmt null rate is {null_rate:.4f}")

    # ----------------------------------------------------------------
    # Check 8: card1 null rate is below 5%
    # ----------------------------------------------------------------
    if "card1" in df.columns:
        null_rate = df["card1"].isnull().mean()
        if null_rate > 0.05:
            warnings.append(
                f"card1 null rate {null_rate:.4f} exceeds 5% threshold"
            )
        else:
            logger.info(f"Check 8 passed: card1 null rate is {null_rate:.4f}")
    
    # ----------------------------------------------------------------
    # Check 9: ProductCD cardinality is within expected range
    # ----------------------------------------------------------------
    if "ProductCD" in df.columns:
        cardinality = df["ProductCD"].nunique()
        if cardinality < 1 or cardinality > 20:
            warnings.append(
                f"ProductCD cardinality {cardinality} is outside expected range 1 to 20"
            )
        else:
            logger.info(f"Check 9 passed: ProductCD cardinality is {cardinality}")

    # ----------------------------------------------------------------
    # Check 10: TransactionDT is non-negative and increasing
    # ----------------------------------------------------------------
    if "TransactionDT" in df.columns:
        if (df["TransactionDT"] < 0).any():
            errors.append("TransactionDT contains negative values")
        else:
            logger.info("Check 10 passed: TransactionDT values are non-negative")

    # ----------------------------------------------------------------
    # Check 11: Identity join rate is within expected range
    # If less than 1% of transactions have identity data something is wrong
    # ----------------------------------------------------------------
    if "id_01" in df.columns:
        identity_match_rate = df["id_01"].notnull().mean()
        if identity_match_rate < 0.01:
            errors.append(
                f"Identity match rate {identity_match_rate:.4f} is suspiciously low, "
                f"check the merge on TransactionID"
            )
        else:
            logger.info(
                f"Check 11 passed: Identity match rate is {identity_match_rate:.4f}"
            )
    
    # ----------------------------------------------------------------
    # Write validation report
    # ----------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(report_dir) / f"validation_report_{timestamp}.txt"
    
    with open(report_path, "w") as f:
        f.write(f"Validation Report\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Dataset shape: {df.shape}\n\n")

        if errors:
            f.write("ERRORS (pipeline will halt):\n")
            for error in errors:
                f.write(f"  - {error}\n")
        else:
            f.write("ERRORS: None\n")

        f.write("\n")

        if warnings:
            f.write("WARNINGS (pipeline continues):\n")
            for warning in warnings:
                f.write(f"  - {warning}\n")
        else:
            f.write("WARNINGS: None\n")

        f.write("\n")
        f.write("Column null rates:\n")
        null_rates = df.isnull().mean().sort_values(ascending=False)
        for col, rate in null_rates.head(20).items():
            f.write(f"  {col}: {rate:.4f}\n")

    logger.info(f"Validation report written to {report_path}")
    
    # ----------------------------------------------------------------
    # Halt pipeline if there are errors
    # ----------------------------------------------------------------
    if errors:
        error_summary = "\n".join(errors)
        raise ValueError(
            f"Data validation failed. Pipeline halted.\n"
            f"Errors:\n{error_summary}\n"
            f"See full report at {report_path}"
        )

    if warnings:
        for warning in warnings:
            logger.warning(warning)

    logger.info("All critical validation checks passed.")
    return True


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    
    logging.basicConfig(level=logging.INFO)

    from src.ingestion.loader import load_raw_data
    
    df = load_raw_data("data/raw")
    validate_raw_data(df)