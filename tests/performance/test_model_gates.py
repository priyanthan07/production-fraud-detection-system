import pytest
import pandas as pd
import numpy as np
from src.training.evaluator import evaluate_model
from src.training.threshold_optimizer import find_optimal_threshold

# Minimum acceptable metrics for the model to pass quality gates
# These are conservative thresholds any reasonable model should exceed
MIN_AUC_ROC = 0.85
MIN_AUC_PR = 0.50
MIN_RECALL = 0.60
MIN_PRECISION = 0.30

def make_good_predictions():
    """
        Simulate a reasonably good fraud detection model.
        Fraud rate is 3.5% matching real dataset distribution.
    """
    
    np.random.seed(42)
    n = 10000
    fraud_rate = 0.035
    
    y_true = np.random.binomial(1, fraud_rate, n)
    
    # Simulate a model that is better than random
    # Fraud cases get higher predicted probabilities
    y_pred_proba = np.where(
        y_true == 1,
        np.random.beta(5, 2, n),   # fraud: skewed toward higher probs
        np.random.beta(1, 5, n),   # legit: skewed toward lower probs
    )

    return y_true, y_pred_proba

def make_bad_predictions():
    """
        Simulate a bad model that is barely better than random.
    """
    np.random.seed(42)
    n = 10000
    fraud_rate = 0.035

    y_true = np.random.binomial(1, fraud_rate, n)
    y_pred_proba = np.random.uniform(0, 1, n)

    return y_true, y_pred_proba

# ----------------------------------------------------------------
# Evaluator tests
# ----------------------------------------------------------------

def test_evaluate_model_returns_all_metrics():
    y_true, y_pred_proba = make_good_predictions()
    metrics = evaluate_model(y_true, y_pred_proba, threshold=0.5)

    required_keys = [
        "auc_roc", "auc_pr", "precision", "recall",
        "f1", "threshold", "predicted_fraud_rate", "actual_fraud_rate"
    ]
    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"


def test_evaluate_model_metrics_in_valid_range():
    y_true, y_pred_proba = make_good_predictions()
    metrics = evaluate_model(y_true, y_pred_proba, threshold=0.5)

    assert 0.0 <= metrics["auc_roc"] <= 1.0
    assert 0.0 <= metrics["auc_pr"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0


def test_evaluate_model_requires_both_classes():
    y_true_all_zeros = np.zeros(100)
    y_pred_proba = np.random.uniform(0, 1, 100)

    with pytest.raises(ValueError, match="both classes"):
        evaluate_model(y_true_all_zeros, y_pred_proba)


def test_good_model_beats_minimum_thresholds():
    """
    A model with genuine discriminative ability should
    exceed our minimum quality gates.
    """
    y_true, y_pred_proba = make_good_predictions()
    threshold = find_optimal_threshold(y_true, y_pred_proba, strategy="f1")
    metrics = evaluate_model(y_true, y_pred_proba, threshold=threshold)

    assert metrics["auc_roc"] >= MIN_AUC_ROC, (
        f"AUC-ROC {metrics['auc_roc']} below minimum {MIN_AUC_ROC}"
    )
    assert metrics["auc_pr"] >= MIN_AUC_PR, (
        f"AUC-PR {metrics['auc_pr']} below minimum {MIN_AUC_PR}"
    )
    assert metrics["recall"] >= MIN_RECALL, (
        f"Recall {metrics['recall']} below minimum {MIN_RECALL}"
    )


def test_random_model_fails_quality_gates():
    """
    A random model should not pass our quality gates.
    This verifies the gates are meaningful.
    """
    y_true, y_pred_proba = make_bad_predictions()
    threshold = find_optimal_threshold(y_true, y_pred_proba, strategy="f1")
    metrics = evaluate_model(y_true, y_pred_proba, threshold=threshold)

    # A random model should fail at least one gate
    passes_all = (
        metrics["auc_roc"] >= MIN_AUC_ROC and
        metrics["auc_pr"] >= MIN_AUC_PR and
        metrics["recall"] >= MIN_RECALL
    )
    assert not passes_all, (
        "Random model passed all quality gates. Gates are too lenient."
    )

# ----------------------------------------------------------------
# Evaluator tests
# ----------------------------------------------------------------

def test_evaluate_model_returns_all_metrics():
    y_true, y_pred_proba = make_good_predictions()
    metrics = evaluate_model(y_true, y_pred_proba, threshold=0.5)

    required_keys = [
        "auc_roc", "auc_pr", "precision", "recall",
        "f1", "threshold", "predicted_fraud_rate", "actual_fraud_rate"
    ]
    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"


def test_evaluate_model_metrics_in_valid_range():
    y_true, y_pred_proba = make_good_predictions()
    metrics = evaluate_model(y_true, y_pred_proba, threshold=0.5)

    assert 0.0 <= metrics["auc_roc"] <= 1.0
    assert 0.0 <= metrics["auc_pr"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0


def test_evaluate_model_requires_both_classes():
    y_true_all_zeros = np.zeros(100)
    y_pred_proba = np.random.uniform(0, 1, 100)

    with pytest.raises(ValueError, match="both classes"):
        evaluate_model(y_true_all_zeros, y_pred_proba)


def test_good_model_beats_minimum_thresholds():
    """
    A model with genuine discriminative ability should
    exceed our minimum quality gates.
    """
    y_true, y_pred_proba = make_good_predictions()
    threshold = find_optimal_threshold(y_true, y_pred_proba, strategy="f1")
    metrics = evaluate_model(y_true, y_pred_proba, threshold=threshold)

    assert metrics["auc_roc"] >= MIN_AUC_ROC, (
        f"AUC-ROC {metrics['auc_roc']} below minimum {MIN_AUC_ROC}"
    )
    assert metrics["auc_pr"] >= MIN_AUC_PR, (
        f"AUC-PR {metrics['auc_pr']} below minimum {MIN_AUC_PR}"
    )
    assert metrics["recall"] >= MIN_RECALL, (
        f"Recall {metrics['recall']} below minimum {MIN_RECALL}"
    )


def test_random_model_fails_quality_gates():
    """
    A random model should not pass our quality gates.
    This verifies the gates are meaningful.
    """
    y_true, y_pred_proba = make_bad_predictions()
    threshold = find_optimal_threshold(y_true, y_pred_proba, strategy="f1")
    metrics = evaluate_model(y_true, y_pred_proba, threshold=threshold)

    # A random model should fail at least one gate
    passes_all = (
        metrics["auc_roc"] >= MIN_AUC_ROC and
        metrics["auc_pr"] >= MIN_AUC_PR and
        metrics["recall"] >= MIN_RECALL
    )
    assert not passes_all, (
        "Random model passed all quality gates. Gates are too lenient."
    )