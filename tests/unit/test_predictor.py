"""
Unit tests for FraudPredictor.

All external dependencies (MLflow, Redis) are mocked.
Tests cover: feature selection, stateless feature computation,
Redis feature injection, prediction output correctness, threshold behavior.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def make_minimal_transaction():
    from src.inference.schemas import TransactionInput

    return TransactionInput(
        TransactionID=1001,
        TransactionDT=86400,
        TransactionAmt=100.0,
    )


def make_predictor_with_mocks(fraud_prob: float = 0.2):
    """
    Build a FraudPredictor with all external dependencies mocked.
    The mock model always returns fraud_prob as its positive class probability.
    """
    from src.inference.predictor import FraudPredictor

    predictor = FraudPredictor()

    # Mock the sklearn model
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[1 - fraud_prob, fraud_prob]])
    predictor.model = mock_model

    # Mock encodings and feature columns
    predictor.encodings = {
        "ProductCD": {"global_mean": 0.035, "category_means": {"W": 0.02, "H": 0.05}},
        "card6": {"global_mean": 0.035, "category_means": {"debit": 0.03, "credit": 0.04}},
    }
    predictor.feature_columns = ["TransactionAmt", "hour_of_day", "ProductCD_encoded"]

    predictor.threshold = 0.5
    predictor.model_version = "test_v1"
    predictor._loaded = True
    predictor._redis_ok = True
    predictor.config = {}

    return predictor


# ================================================================
# _select_model_features
# ================================================================


def test_select_model_features_returns_exact_columns():
    from src.inference.predictor import FraudPredictor

    p = FraudPredictor()
    p.feature_columns = ["a", "b", "c"]

    df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
    result = p._select_model_features(df)

    assert list(result.columns) == ["a", "b", "c"]
    assert "d" not in result.columns


def test_select_model_features_fills_missing_with_nan():
    from src.inference.predictor import FraudPredictor

    p = FraudPredictor()
    p.feature_columns = ["a", "b", "missing_col"]

    df = pd.DataFrame({"a": [1], "b": [2]})
    result = p._select_model_features(df)

    assert "missing_col" in result.columns
    assert np.isnan(result["missing_col"].iloc[0])


def test_select_model_features_preserves_column_order():
    from src.inference.predictor import FraudPredictor

    p = FraudPredictor()
    p.feature_columns = ["c", "a", "b"]

    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    result = p._select_model_features(df)

    assert list(result.columns) == ["c", "a", "b"]


# ================================================================
# _run_stateless_features
# ================================================================


def test_stateless_features_produces_time_columns():
    p = make_predictor_with_mocks()
    df = pd.DataFrame(
        [
            {
                "TransactionID": 1,
                "TransactionDT": 86400 + 3600 * 14,  # 2pm on day 1
                "TransactionAmt": 50.0,
                "card1": 999,
                "P_emaildomain": "gmail.com",
            }
        ]
    )
    result = p._run_stateless_features(df)
    assert "hour_of_day" in result.columns
    assert "is_night_transaction" in result.columns
    assert "is_weekend" in result.columns


def test_stateless_features_drops_stateful_time_cols():
    """
    FIX 2: time_since_last_txn_card1 and days_since_card_first_seen
    must NOT be in the result — they are wrong on 1 row and are
    provided by Redis instead.
    """
    p = make_predictor_with_mocks()
    df = pd.DataFrame(
        [
            {
                "TransactionID": 1,
                "TransactionDT": 86400,
                "TransactionAmt": 50.0,
                "card1": 999,
                "P_emaildomain": None,
            }
        ]
    )
    result = p._run_stateless_features(df)
    assert "time_since_last_txn_card1" not in result.columns
    assert "days_since_card_first_seen" not in result.columns


def test_stateless_features_applies_encoding():
    p = make_predictor_with_mocks()
    df = pd.DataFrame(
        [
            {
                "TransactionID": 1,
                "TransactionDT": 86400,
                "TransactionAmt": 50.0,
                "card1": 999,
                "ProductCD": "W",
                "card6": "debit",
                "P_emaildomain": None,
            }
        ]
    )
    result = p._run_stateless_features(df)
    assert "ProductCD_encoded" in result.columns
    assert abs(result["ProductCD_encoded"].iloc[0] - 0.02) < 1e-6


def test_stateless_features_unseen_category_uses_global_mean():
    p = make_predictor_with_mocks()
    df = pd.DataFrame(
        [
            {
                "TransactionID": 1,
                "TransactionDT": 86400,
                "TransactionAmt": 50.0,
                "card1": 999,
                "ProductCD": "UNSEEN_XYZ",
                "P_emaildomain": None,
            }
        ]
    )
    result = p._run_stateless_features(df)
    # Unseen category should fall back to global mean
    assert abs(result["ProductCD_encoded"].iloc[0] - 0.035) < 1e-6


# ================================================================
# predict_single — output correctness
# ================================================================


def test_predict_single_returns_correct_transaction_id():
    p = make_predictor_with_mocks(fraud_prob=0.2)
    txn = make_minimal_transaction()

    with patch("src.inference.predictor.feature_store") as mock_fs:
        mock_fs.get_card_features.return_value = {
            "card1_count_1hr": 0,
            "card1_amt_sum_1hr": 0.0,
            "card1_count_24hr": 0,
            "card1_amt_sum_24hr": 0.0,
            "card1_count_7day": 0,
            "card1_amt_sum_7day": 0.0,
            "card1_txn_count": 0,
            "card1_amt_mean": float("nan"),
            "card1_amt_std": float("nan"),
            "card1_amt_deviation": float("nan"),
            "card1_amt_zscore": 0.0,
            "days_since_card_first_seen": 0.0,
            "time_since_last_txn_card1": -1.0,
        }
        mock_fs.get_email_features.return_value = {}
        mock_fs.update.return_value = None
        mock_fs.log_scored_features.return_value = None
        mock_fs.log_raw_transaction.return_value = None
        mock_fs.log_prediction_for_drift.return_value = None

        result = p.predict_single(txn)

    assert result.TransactionID == 1001


def test_predict_single_fraud_probability_range():
    p = make_predictor_with_mocks(fraud_prob=0.75)
    txn = make_minimal_transaction()

    with patch("src.inference.predictor.feature_store") as mock_fs:
        mock_fs.get_card_features.return_value = {}
        mock_fs.get_email_features.return_value = {}
        mock_fs.update.return_value = None
        mock_fs.log_scored_features.return_value = None
        mock_fs.log_raw_transaction.return_value = None
        mock_fs.log_prediction_for_drift.return_value = None

        result = p.predict_single(txn)

    assert 0.0 <= result.fraud_probability <= 1.0


def test_predict_single_below_threshold_not_fraud():
    p = make_predictor_with_mocks(fraud_prob=0.3)
    p.threshold = 0.5
    txn = make_minimal_transaction()

    with patch("src.inference.predictor.feature_store") as mock_fs:
        mock_fs.get_card_features.return_value = {}
        mock_fs.get_email_features.return_value = {}
        mock_fs.update.return_value = None
        mock_fs.log_scored_features.return_value = None
        mock_fs.log_raw_transaction.return_value = None
        mock_fs.log_prediction_for_drift.return_value = None

        result = p.predict_single(txn)

    assert result.is_fraud is False


def test_predict_single_above_threshold_is_fraud():
    p = make_predictor_with_mocks(fraud_prob=0.8)
    p.threshold = 0.5
    txn = make_minimal_transaction()

    with patch("src.inference.predictor.feature_store") as mock_fs:
        mock_fs.get_card_features.return_value = {}
        mock_fs.get_email_features.return_value = {}
        mock_fs.update.return_value = None
        mock_fs.log_scored_features.return_value = None
        mock_fs.log_raw_transaction.return_value = None
        mock_fs.log_prediction_for_drift.return_value = None

        result = p.predict_single(txn)

    assert result.is_fraud is True


def test_predict_single_redis_update_called_after_prediction():
    """
    Redis update must be called AFTER predict_proba — not before.
    This ensures the current transaction does not influence its own features.
    """
    p = make_predictor_with_mocks(fraud_prob=0.2)
    txn = make_minimal_transaction()

    call_order = []

    with patch("src.inference.predictor.feature_store") as mock_fs:
        mock_fs.get_card_features.return_value = {}
        mock_fs.get_email_features.return_value = {}
        mock_fs.update.side_effect = lambda **kwargs: call_order.append("redis_update")
        mock_fs.log_scored_features.return_value = None
        mock_fs.log_raw_transaction.return_value = None
        mock_fs.log_prediction_for_drift.return_value = None

        p.model.predict_proba.side_effect = lambda x: (
            call_order.append("predict_proba") or np.array([[0.8, 0.2]])
        )

        p.predict_single(txn)

    assert call_order.index("predict_proba") < call_order.index("redis_update"), (
        "predict_proba must be called before redis_update"
    )


def test_predict_single_raises_if_not_loaded():
    from src.inference.predictor import FraudPredictor

    p = FraudPredictor()
    # _loaded defaults to False

    with pytest.raises(RuntimeError, match="not loaded"):
        p.predict_single(make_minimal_transaction())


# ================================================================
# predict_batch
# ================================================================


def test_predict_batch_returns_in_input_order():
    """Results must be returned in the original input order, not chronological."""
    p = make_predictor_with_mocks(fraud_prob=0.2)

    from src.inference.schemas import TransactionInput

    txns = [
        TransactionInput(TransactionID=3, TransactionDT=93600, TransactionAmt=30.0),
        TransactionInput(TransactionID=1, TransactionDT=86400, TransactionAmt=10.0),
        TransactionInput(TransactionID=2, TransactionDT=90000, TransactionAmt=20.0),
    ]

    with patch("src.inference.predictor.feature_store") as mock_fs:
        mock_fs.get_card_features.return_value = {}
        mock_fs.get_email_features.return_value = {}
        mock_fs.update.return_value = None
        mock_fs.log_scored_features.return_value = None
        mock_fs.log_raw_transaction.return_value = None
        mock_fs.log_prediction_for_drift.return_value = None

        results = p.predict_batch(txns)

    assert [r.TransactionID for r in results] == [3, 1, 2]


def test_predict_batch_empty_returns_empty():
    p = make_predictor_with_mocks()
    with patch("src.inference.predictor.feature_store"):
        results = p.predict_batch([])
    assert results == []
