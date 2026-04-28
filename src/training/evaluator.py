import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_model(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute all evaluation metrics for a fraud detection model.

    Args:
        y_true: actual fraud labels, array of 0s and 1s
        y_pred_proba: predicted fraud probabilities between 0 and 1
        threshold: decision threshold to convert probabilities to labels

    Returns:
        dictionary of metric names to values
    """

    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true must contain both classes (0 and 1). Check your validation split.")

    # Convert probabilities to binary predictions using threshold
    y_pred = (y_pred_proba >= threshold).astype(int)

    # AUC-ROC: measures overall discriminative ability
    # Higher is better. 0.5 = random, 1.0 = perfect
    auc_roc = roc_auc_score(y_true, y_pred_proba)

    # AUC-PR: area under precision recall curve
    # More informative than AUC-ROC for imbalanced datasets
    # Because it focuses on the positive (fraud) class
    auc_pr = average_precision_score(y_true, y_pred_proba)

    # Precision: of all flagged fraud how many are actually fraud
    # zero_division=0 means if no positives predicted return 0
    precision = precision_score(y_true, y_pred, zero_division=0)

    # Recall: of all actual fraud how many did we catch
    recall = recall_score(y_true, y_pred, zero_division=0)

    # F1: harmonic mean of precision and recall
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Fraud rate in predictions vs actual
    predicted_fraud_rate = y_pred.mean()
    actual_fraud_rate = y_true.mean()

    metrics = {
        "auc_roc": round(float(auc_roc), 4),
        "auc_pr": round(float(auc_pr), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "threshold": round(float(threshold), 4),
        "predicted_fraud_rate": round(float(predicted_fraud_rate), 4),
        "actual_fraud_rate": round(float(actual_fraud_rate), 4),
    }

    logger.info("Evaluation metrics:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value}")

    return metrics


def compute_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> pd.DataFrame:
    """
    Compute precision and recall at every threshold.
    Useful for visualizing the tradeoff and choosing threshold.

    Returns dataframe with columns: threshold, precision, recall, f1
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    with np.errstate(invalid="ignore"):
        f1_scores = np.where(
            (precisions[:-1] + recalls[:-1]) > 0,
            2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
            0.0,
        )

    curve_df = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precisions[:-1],
            "recall": recalls[:-1],
            "f1": f1_scores,
        }
    )

    return curve_df
