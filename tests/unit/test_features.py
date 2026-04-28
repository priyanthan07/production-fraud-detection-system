import pandas as pd

from src.features.categorical_encoder import (
    apply_target_encoder,
    fit_target_encoder,
)
from src.features.time_features import compute_time_features
from src.features.user_aggregations import compute_user_aggregations
from src.features.velocity_features import compute_velocity_features


def make_sample_df():
    """Minimal valid dataframe matching the real dataset structure."""
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6],
            "isFraud": [0, 0, 1, 0, 0, 1],
            "TransactionDT": [86400, 90000, 93600, 180000, 266400, 604800],
            "TransactionAmt": [10.0, 20.0, 200.0, 15.0, 25.0, 300.0],
            "ProductCD": ["W", "W", "H", "W", "C", "H"],
            "card1": [1001, 1001, 1001, 2001, 2001, 1001],
            "card4": ["visa", "visa", "mastercard", "visa", "visa", "mastercard"],
            "card6": ["debit", "debit", "credit", "debit", "debit", "credit"],
            "P_emaildomain": [
                "gmail.com",
                "gmail.com",
                "yahoo.com",
                "gmail.com",
                None,
                "yahoo.com",
            ],
            "R_emaildomain": [None, None, None, None, None, None],
            "M1": ["T", "T", "F", "T", "T", "F"],
            "M2": ["T", "F", "T", "T", "F", "T"],
            "M3": ["T", "T", "T", "F", "T", "F"],
            "M4": ["M0", "M1", "M2", "M0", "M1", "M2"],
            "M5": ["F", "T", "F", "F", "T", "F"],
            "M6": ["T", "T", "F", "T", "T", "F"],
            "M7": [None, None, None, None, None, None],
            "M8": [None, None, None, None, None, None],
            "M9": [None, None, None, None, None, None],
        }
    )


# ----------------------------------------------------------------
# Time feature tests
# ----------------------------------------------------------------


def test_time_features_columns_created():
    df = make_sample_df()
    result = compute_time_features(df)
    expected_cols = [
        "hour_of_day",
        "day_of_week",
        "days_since_start",
        "time_since_last_txn_card1",
        "days_since_card_first_seen",
        "is_night_transaction",
        "is_weekend",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_hour_of_day_range():
    df = make_sample_df()
    result = compute_time_features(df)
    assert result["hour_of_day"].between(0, 23).all()


def test_day_of_week_range():
    df = make_sample_df()
    result = compute_time_features(df)
    assert result["day_of_week"].between(0, 6).all()


def test_days_since_card_first_seen_non_negative():
    df = make_sample_df()
    result = compute_time_features(df)
    assert (result["days_since_card_first_seen"] >= 0).all()


def test_is_night_transaction_binary():
    df = make_sample_df()
    result = compute_time_features(df)
    assert result["is_night_transaction"].isin([0, 1]).all()


def test_is_weekend_binary():
    df = make_sample_df()
    result = compute_time_features(df)
    assert result["is_weekend"].isin([0, 1]).all()


# ----------------------------------------------------------------
# Velocity feature tests
# ----------------------------------------------------------------
def test_velocity_columns_created():
    df = make_sample_df()
    result = compute_velocity_features(df)
    expected_cols = [
        "card1_count_1hr",
        "card1_amt_sum_1hr",
        "card1_count_24hr",
        "card1_amt_sum_24hr",
        "card1_count_7day",
        "card1_amt_sum_7day",
        "P_emaildomain_count_1hr",
        "P_emaildomain_amt_sum_1hr",
        "P_emaildomain_count_24hr",
        "P_emaildomain_amt_sum_24hr",
        "P_emaildomain_count_7day",
        "P_emaildomain_amt_sum_7day",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_velocity_counts_non_negative():
    df = make_sample_df()
    result = compute_velocity_features(df)
    count_cols = [c for c in result.columns if "count" in c]
    for col in count_cols:
        valid = result[col].dropna()
        assert (valid >= 0).all(), f"Negative count in {col}"


def test_first_transaction_velocity_is_zero():
    """
    The first transaction of each card should have count 0
    because there are no prior transactions in any window.
    """
    df = make_sample_df()
    result = compute_velocity_features(df)
    result = result.sort_values("TransactionDT")
    first_card1001 = result[result["card1"] == 1001].iloc[0]
    assert first_card1001["card1_count_1hr"] == 0
    assert first_card1001["card1_count_24hr"] == 0
    assert first_card1001["card1_count_7day"] == 0


# ----------------------------------------------------------------
# User aggregation tests
# ----------------------------------------------------------------


def test_user_aggregation_columns_created():
    df = make_sample_df()
    result = compute_user_aggregations(df)
    expected_cols = [
        "card1_amt_mean",
        "card1_amt_std",
        "card1_txn_count",
        "card1_amt_deviation",
        "card1_amt_zscore",
        "email_amt_mean",
        "email_amt_std",
        "email_amt_deviation",
        "email_amt_zscore",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_first_transaction_per_card_has_nan_mean():
    """
    First transaction per card has no prior history
    so historical mean must be NaN.
    """
    df = make_sample_df()
    result = compute_user_aggregations(df)
    result = result.sort_values("TransactionDT").reset_index(drop=True)
    first_card1001 = result[result["card1"] == 1001].iloc[0]
    assert pd.isna(first_card1001["card1_amt_mean"])


def test_txn_count_non_negative():
    df = make_sample_df()
    result = compute_user_aggregations(df)
    valid_counts = result["card1_txn_count"].dropna()
    assert (valid_counts >= 0).all()


def test_no_future_data_leakage():
    """
    Count for first transaction of a card must be 0.
    If greater than 0 then future data was used.
    """
    df = make_sample_df()
    result = compute_user_aggregations(df)
    result = result.sort_values("TransactionDT")
    first_txn = result[result["card1"] == 2001].iloc[0]
    count = first_txn["card1_txn_count"]
    assert pd.isna(count) or count == 0


def test_null_email_rows_have_nan_email_features():
    """
    Rows with null P_emaildomain should have NaN email aggregations.
    """
    df = make_sample_df()
    result = compute_user_aggregations(df)
    null_email_rows = result[result["P_emaildomain"].isna()]
    assert null_email_rows["email_amt_mean"].isna().all()


def test_email_zscore_exists():
    df = make_sample_df()
    result = compute_user_aggregations(df)
    assert "email_amt_zscore" in result.columns


# ----------------------------------------------------------------
# Categorical encoder tests
# ----------------------------------------------------------------


def test_target_encoder_returns_tuple():
    df = make_sample_df()
    result = fit_target_encoder(df, ["ProductCD", "card4"])
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_target_encoder_produces_encoded_columns():
    df = make_sample_df()
    df_encoded, encodings = fit_target_encoder(df, ["ProductCD"])
    assert "ProductCD_encoded" in df_encoded.columns


def test_target_encoder_global_mean_correct():
    df = make_sample_df()
    df_encoded, encodings = fit_target_encoder(df, ["ProductCD"], target_col="isFraud")
    expected_global_mean = df["isFraud"].mean()
    assert abs(encodings["ProductCD"]["global_mean"] - expected_global_mean) < 1e-6


def test_apply_encoder_creates_encoded_columns():
    df = make_sample_df()
    df_encoded, encodings = fit_target_encoder(df, ["ProductCD"])
    result = apply_target_encoder(df, encodings)
    assert "ProductCD_encoded" in result.columns


def test_unseen_category_falls_back_to_global_mean():
    df = make_sample_df()
    df_encoded, encodings = fit_target_encoder(df, ["ProductCD"])
    test_df = df.copy()
    test_df.loc[0, "ProductCD"] = "UNSEEN_CATEGORY"
    result = apply_target_encoder(test_df, encodings)
    global_mean = encodings["ProductCD"]["global_mean"]
    assert abs(result.loc[0, "ProductCD_encoded"] - global_mean) < 1e-6


def test_encoded_column_names_match_between_train_and_inference():
    """
    Column names created during training must exactly match
    column names created at inference time.
    This ensures the model sees the same feature names in both phases.
    """
    df = make_sample_df()
    df_encoded, encodings = fit_target_encoder(df, ["ProductCD", "card4"])
    inference_df = apply_target_encoder(df, encodings)

    training_cols = sorted([c for c in df_encoded.columns if c.endswith("_encoded")])
    inference_cols = sorted([c for c in inference_df.columns if c.endswith("_encoded")])

    assert training_cols == inference_cols, (
        f"Training columns {training_cols} do not match inference columns {inference_cols}"
    )


def test_original_dataframe_not_modified():
    """
    fit_target_encoder must not modify the dataframe passed in.
    """
    df = make_sample_df()
    original_cols = list(df.columns)
    fit_target_encoder(df, ["ProductCD"])
    assert list(df.columns) == original_cols
