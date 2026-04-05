import pytest
import pandas as pd
from src.ingestion.validator import validate_raw_data

def make_valid_df():
    """Create a minimal valid dataframe for testing."""
    return pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5],
        "isFraud": [0, 0, 0, 0, 1],
        "TransactionDT": [100, 200, 300, 400, 500],
        "TransactionAmt": [10.0, 20.0, 30.0, 40.0, 50.0],
        "ProductCD": ["W", "H", "C", "S", "R"],
        "card1": [1001, 1002, 1003, 1004, 1005],
        "card2": [200, 201, 202, 203, 204],
        "card3": [150, 150, 150, 150, 150],
        "card4": ["visa", "mastercard", "visa", "visa", "mastercard"],
        "card5": [226, 226, 226, 226, 226],
        "card6": ["debit", "credit", "debit", "debit", "credit"],
        "P_emaildomain": ["gmail.com", "yahoo.com", None, "gmail.com", "hotmail.com"],
        "R_emaildomain": [None, None, None, None, None],
    })
    
def test_valid_dataframe_passes():
    df = make_valid_df()
    result = validate_raw_data(df, report_dir="data/validation_reports/tests")
    assert result is True
    
def test_missing_column_raises_error():
    df = make_valid_df()
    df = df.drop(columns=["TransactionAmt"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_raw_data(df, report_dir="data/validation_reports/tests")

def test_duplicate_transaction_ids_raises_error():
    df = make_valid_df()
    df.loc[1, "TransactionID"] = df.loc[0, "TransactionID"]
    with pytest.raises(ValueError, match="duplicate TransactionIDs"):
        validate_raw_data(df, report_dir="data/validation_reports/tests")

def test_invalid_fraud_label_raises_error():
    df = make_valid_df()
    df.loc[0, "isFraud"] = 5
    with pytest.raises(ValueError, match="isFraud column contains values"):
        validate_raw_data(df, report_dir="data/validation_reports/tests")

def test_negative_transaction_amount_raises_error():
    df = make_valid_df()
    df.loc[0, "TransactionAmt"] = -10.0
    with pytest.raises(ValueError, match="negative TransactionAmt"):
        validate_raw_data(df, report_dir="data/validation_reports/tests")

def test_empty_dataframe_raises_error():
    df = make_valid_df().iloc[0:0]
    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_raw_data(df, report_dir="data/validation_reports/tests")