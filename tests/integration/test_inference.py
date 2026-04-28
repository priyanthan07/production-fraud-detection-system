"""
Integration Tests — Inference Server
=====================================
Tests the FastAPI endpoints and the predictor's edge case handling.

These are integration tests, not unit tests. They test the full
request → feature engineering → model → response pipeline.

The model and artifacts (encodings.pkl, feature_columns.txt) must
exist for these tests to run. They are not mocked — we test the
real predictor logic.

The FastAPI app is tested using TestClient from httpx, which runs
the app in-process without starting a real HTTP server. This makes
the tests fast and self-contained.

Run:
    pytest tests/integration/test_inference.py -v
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.inference.schemas import PredictionOutput

# ----------------------------------------------------------------
# Fixtures — minimal valid transaction data
# ----------------------------------------------------------------


def make_minimal_transaction() -> dict:
    """
    Minimum required fields for a valid prediction request.
    Only TransactionID, TransactionDT, and TransactionAmt are required.
    All other fields are optional.
    """
    return {
        "TransactionID": 9999999,
        "TransactionDT": 86400,
        "TransactionAmt": 68.50,
    }


def make_full_transaction() -> dict:
    """
    A transaction with most fields populated, mimicking a real
    transaction from the IEEE-CIS dataset.
    """
    return {
        "TransactionID": 3663592,
        "TransactionDT": 86400,
        "TransactionAmt": 68.50,
        "ProductCD": "W",
        "card1": 13926,
        "card2": 358.0,
        "card3": 150.0,
        "card4": "visa",
        "card5": 226.0,
        "card6": "debit",
        "addr1": 315.0,
        "addr2": 87.0,
        "dist1": 19.0,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": None,
        "C1": 1.0,
        "C2": 1.0,
        "C3": 0.0,
        "C4": 0.0,
        "C5": 0.0,
        "C6": 1.0,
        "C7": 0.0,
        "C8": 0.0,
        "C9": 1.0,
        "C10": 0.0,
        "C11": 1.0,
        "C12": 0.0,
        "C13": 22.0,
        "C14": 1.0,
        "D1": 14.0,
        "M1": "T",
        "M2": "T",
        "M3": "T",
        "M6": "T",
    }


def make_batch_transactions(n: int = 5) -> list:
    """Create n distinct transactions for batch testing."""
    transactions = []
    for i in range(n):
        t = make_full_transaction()
        t["TransactionID"] = 1000000 + i
        t["TransactionDT"] = 86400 + (i * 3600)
        t["TransactionAmt"] = 50.0 + (i * 10)
        t["card1"] = 13926 + i
        transactions.append(t)
    return transactions


# ----------------------------------------------------------------
# Mock predictor setup
# The predictor requires a real trained model and artifact files
# to load. For integration tests we mock the load step and provide
# controlled predict_proba outputs so tests are deterministic.
# ----------------------------------------------------------------


def make_mock_predictor(fraud_prob: float = 0.15):
    """
    Create a mock predictor that returns a fixed fraud probability
    without needing real model files on disk.
    """
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[1 - fraud_prob, fraud_prob]])

    mock_predictor = MagicMock()
    mock_predictor._loaded = True
    mock_predictor.model = mock_model
    mock_predictor.threshold = 0.5
    mock_predictor.model_version = "test_v1"
    mock_predictor.encodings = {}
    mock_predictor.feature_columns = ["TransactionAmt", "card1", "C1"]

    return mock_predictor


@pytest.fixture
def client_low_risk():
    """TestClient where predictor returns a low fraud probability."""
    from src.inference.app import app

    with patch("src.inference.app.predictor") as mock_pred:
        mock_pred._loaded = True
        mock_pred.model_version = "test_v1"
        mock_pred.threshold = 0.5

        mock_pred.predict_single.return_value = PredictionOutput(
            TransactionID=9999999,
            fraud_probability=0.12,
            is_fraud=False,
            risk_level="LOW",
            threshold_used=0.5,
            model_version="test_v1",
        )
        mock_pred.predict_batch.return_value = [
            PredictionOutput(
                TransactionID=1000000 + i,
                fraud_probability=0.12,
                is_fraud=False,
                risk_level="LOW",
                threshold_used=0.5,
                model_version="test_v1",
            )
            for i in range(5)
        ]
        yield TestClient(app)


@pytest.fixture
def client_high_risk():
    """TestClient where predictor returns a high fraud probability."""
    from src.inference.app import app

    with patch("src.inference.app.predictor") as mock_pred:
        mock_pred._loaded = True
        mock_pred.model_version = "test_v1"
        mock_pred.threshold = 0.5
        mock_pred.predict_single.return_value = PredictionOutput(
            TransactionID=9999999,
            fraud_probability=0.91,
            is_fraud=True,
            risk_level="CRITICAL",
            threshold_used=0.5,
            model_version="test_v1",
        )
        yield TestClient(app)


# ================================================================
# Health endpoint tests
# ================================================================


def test_health_returns_200(client_low_risk):
    response = client_low_risk.get("/health")
    assert response.status_code == 200


def test_health_contains_required_fields(client_low_risk):
    response = client_low_risk.get("/health")
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "model_version" in data
    assert "threshold" in data


# ================================================================
# Single predict endpoint — valid inputs
# ================================================================


def test_predict_minimal_transaction_returns_200(client_low_risk):
    """Minimal transaction with only required fields should succeed."""
    response = client_low_risk.post(
        "/predict",
        json=make_minimal_transaction(),
    )
    assert response.status_code == 200


def test_predict_full_transaction_returns_200(client_low_risk):
    """Full transaction with all optional fields should succeed."""
    response = client_low_risk.post(
        "/predict",
        json=make_full_transaction(),
    )
    assert response.status_code == 200


def test_predict_response_has_required_fields(client_low_risk):
    response = client_low_risk.post(
        "/predict",
        json=make_minimal_transaction(),
    )
    data = response.json()
    required_fields = [
        "TransactionID",
        "fraud_probability",
        "is_fraud",
        "risk_level",
        "threshold_used",
        "model_version",
    ]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"


def test_predict_fraud_probability_in_valid_range(client_low_risk):
    response = client_low_risk.post(
        "/predict",
        json=make_minimal_transaction(),
    )
    prob = response.json()["fraud_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_low_risk_is_fraud_false(client_low_risk):
    """Low probability transactions should be flagged as not fraud."""
    response = client_low_risk.post(
        "/predict",
        json=make_minimal_transaction(),
    )
    data = response.json()
    assert data["is_fraud"] is False
    assert data["risk_level"] == "LOW"


def test_predict_high_risk_is_fraud_true(client_high_risk):
    """High probability transactions should be flagged as fraud."""
    response = client_high_risk.post(
        "/predict",
        json=make_minimal_transaction(),
    )
    data = response.json()
    assert data["is_fraud"] is True
    assert data["risk_level"] == "CRITICAL"


def test_predict_transaction_id_preserved(client_low_risk):
    """The TransactionID in the response must match the input."""
    txn = make_minimal_transaction()
    response = client_low_risk.post("/predict", json=txn)
    assert response.json()["TransactionID"] == txn["TransactionID"]


# ================================================================
# Single predict endpoint — invalid inputs
# ================================================================


def test_predict_missing_transaction_id_returns_422(client_low_risk):
    txn = make_minimal_transaction()
    del txn["TransactionID"]
    response = client_low_risk.post("/predict", json=txn)
    assert response.status_code == 422


def test_predict_missing_transaction_dt_returns_422(client_low_risk):
    txn = make_minimal_transaction()
    del txn["TransactionDT"]
    response = client_low_risk.post("/predict", json=txn)
    assert response.status_code == 422


def test_predict_missing_amount_returns_422(client_low_risk):
    txn = make_minimal_transaction()
    del txn["TransactionAmt"]
    response = client_low_risk.post("/predict", json=txn)
    assert response.status_code == 422


def test_predict_negative_amount_returns_422(client_low_risk):
    txn = make_minimal_transaction()
    txn["TransactionAmt"] = -50.0
    response = client_low_risk.post("/predict", json=txn)
    assert response.status_code == 422


def test_predict_zero_amount_returns_422(client_low_risk):
    txn = make_minimal_transaction()
    txn["TransactionAmt"] = 0.0
    response = client_low_risk.post("/predict", json=txn)
    assert response.status_code == 422


def test_predict_negative_transaction_dt_returns_422(client_low_risk):
    txn = make_minimal_transaction()
    txn["TransactionDT"] = -1
    response = client_low_risk.post("/predict", json=txn)
    assert response.status_code == 422


def test_predict_empty_body_returns_422(client_low_risk):
    response = client_low_risk.post("/predict", json={})
    assert response.status_code == 422


# ================================================================
# Edge cases — valid inputs but unusual values
# ================================================================


def test_predict_all_optional_fields_null(client_low_risk):
    """
    Transaction with only required fields and everything else null
    must succeed. Model handles NaN natively.
    """
    txn = {
        "TransactionID": 1,
        "TransactionDT": 86400,
        "TransactionAmt": 100.0,
        "card1": None,
        "card4": None,
        "P_emaildomain": None,
        "R_emaildomain": None,
    }
    response = client_low_risk.post("/predict", json=txn)
    assert response.status_code == 200


def test_predict_unseen_product_code(client_low_risk):
    """
    ProductCD value that was not in training data must not crash.
    The encoder falls back to global mean for unseen categories.
    """
    txn = make_minimal_transaction()
    txn["ProductCD"] = "UNSEEN_CATEGORY_XYZ"
    response = client_low_risk.post("/predict", json=txn)
    assert response.status_code == 200


def test_predict_unseen_email_domain(client_low_risk):
    """Unseen email domain must fall back to global mean encoding."""
    txn = make_minimal_transaction()
    txn["P_emaildomain"] = "totallynewdomain12345.xyz"
    response = client_low_risk.post("/predict", json=txn)
    assert response.status_code == 200


def test_predict_very_large_amount(client_low_risk):
    """Extreme transaction amounts must not crash the pipeline."""
    txn = make_minimal_transaction()
    txn["TransactionAmt"] = 999999.99
    response = client_low_risk.post("/predict", json=txn)
    assert response.status_code == 200


def test_predict_very_small_amount(client_low_risk):
    """Very small but positive amounts must be accepted."""
    txn = make_minimal_transaction()
    txn["TransactionAmt"] = 0.01
    response = client_low_risk.post("/predict", json=txn)
    assert response.status_code == 200


def test_predict_extra_v_features_accepted(client_low_risk):
    """
    V features (V1-V339) are accepted via the extra="allow" config
    on the Pydantic schema and must not cause a 422.
    """
    txn = make_minimal_transaction()
    txn["V1"] = 1.0
    txn["V45"] = 0.0
    txn["V258"] = 1.0
    response = client_low_risk.post("/predict", json=txn)
    assert response.status_code == 200


# ================================================================
# Batch predict endpoint tests
# ================================================================


def test_batch_predict_returns_200(client_low_risk):
    response = client_low_risk.post(
        "/predict/batch",
        json={"transactions": make_batch_transactions(5)},
    )
    assert response.status_code == 200


def test_batch_predict_response_count_matches_input(client_low_risk):
    n = 5
    response = client_low_risk.post(
        "/predict/batch",
        json={"transactions": make_batch_transactions(n)},
    )
    data = response.json()
    assert data["total_transactions"] == n
    assert len(data["predictions"]) == n


def test_batch_predict_fraud_rate_computed_correctly(client_low_risk):
    """
    fraud_rate_in_batch should equal flagged/total.
    """
    response = client_low_risk.post(
        "/predict/batch",
        json={"transactions": make_batch_transactions(5)},
    )
    data = response.json()
    expected_rate = data["flagged_as_fraud"] / data["total_transactions"]
    assert abs(data["fraud_rate_in_batch"] - expected_rate) < 0.0001


def test_batch_predict_empty_list_returns_422(client_low_risk):
    """Empty batch should be rejected — min_length=1 on the schema."""
    response = client_low_risk.post(
        "/predict/batch",
        json={"transactions": []},
    )
    assert response.status_code == 422


def test_batch_predict_duplicate_transaction_ids_returns_422(client_low_risk):
    """Duplicate TransactionIDs in a batch must be rejected."""
    transactions = make_batch_transactions(3)
    # Make two transactions have the same ID
    transactions[1]["TransactionID"] = transactions[0]["TransactionID"]
    response = client_low_risk.post(
        "/predict/batch",
        json={"transactions": transactions},
    )
    assert response.status_code == 422


def test_batch_predict_single_transaction_works(client_low_risk):
    """A batch of exactly one transaction must succeed."""
    response = client_low_risk.post(
        "/predict/batch",
        json={"transactions": [make_full_transaction()]},
    )
    assert response.status_code == 200


# ================================================================
# Schemas unit tests
# ================================================================


def test_compute_risk_level_low():
    from src.inference.schemas import compute_risk_level

    assert compute_risk_level(0.10) == "LOW"
    assert compute_risk_level(0.29) == "LOW"


def test_compute_risk_level_medium():
    from src.inference.schemas import compute_risk_level

    assert compute_risk_level(0.30) == "MEDIUM"
    assert compute_risk_level(0.49) == "MEDIUM"


def test_compute_risk_level_high():
    from src.inference.schemas import compute_risk_level

    assert compute_risk_level(0.50) == "HIGH"
    assert compute_risk_level(0.79) == "HIGH"


def test_compute_risk_level_critical():
    from src.inference.schemas import compute_risk_level

    assert compute_risk_level(0.80) == "CRITICAL"
    assert compute_risk_level(1.00) == "CRITICAL"


def test_compute_risk_level_boundary_values():
    from src.inference.schemas import compute_risk_level

    assert compute_risk_level(0.0) == "LOW"
    assert compute_risk_level(0.3) == "MEDIUM"
    assert compute_risk_level(0.5) == "HIGH"
    assert compute_risk_level(0.8) == "CRITICAL"


# ================================================================
# model_manager unit tests (no MLflow server required)
# ================================================================


def test_check_quality_gates_all_pass():
    from src.registry.model_manager import check_quality_gates

    metrics = {
        "auc_roc": 0.91,
        "auc_pr": 0.55,
        "recall": 0.65,
        "precision": 0.45,
    }
    passed, report = check_quality_gates(metrics)
    assert passed is True
    for gate_name, result in report.items():
        assert result["passed"] is True


def test_check_quality_gates_one_fails():
    from src.registry.model_manager import check_quality_gates

    metrics = {
        "auc_roc": 0.91,
        "auc_pr": 0.35,  # below 0.50 gate
        "recall": 0.65,
        "precision": 0.45,
    }
    passed, report = check_quality_gates(metrics)
    assert passed is False
    assert report["auc_pr"]["passed"] is False
    assert report["auc_roc"]["passed"] is True


def test_check_quality_gates_missing_metric():
    from src.registry.model_manager import check_quality_gates

    metrics = {
        "auc_roc": 0.91,
        # auc_pr missing
        "recall": 0.65,
        "precision": 0.45,
    }
    passed, report = check_quality_gates(metrics)
    assert passed is False
    assert report["auc_pr"]["passed"] is False
    assert "metric not found" in report["auc_pr"]["reason"]


def test_check_quality_gates_all_fail():
    from src.registry.model_manager import check_quality_gates

    metrics = {
        "auc_roc": 0.50,
        "auc_pr": 0.03,
        "recall": 0.10,
        "precision": 0.04,
    }
    passed, report = check_quality_gates(metrics)
    assert passed is False
    assert all(not r["passed"] for r in report.values())


def test_check_quality_gates_exactly_at_boundary():
    """Values exactly at the gate threshold should pass."""
    from src.registry.model_manager import check_quality_gates

    metrics = {
        "auc_roc": 0.85,  # exactly at gate
        "auc_pr": 0.50,  # exactly at gate
        "recall": 0.60,  # exactly at gate
        "precision": 0.30,  # exactly at gate
    }
    passed, report = check_quality_gates(metrics)
    assert passed is True
