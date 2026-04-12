import numpy as np
import pandas as pd
import logging
from src.training.evaluator import compute_precision_recall_curve, evaluate_model

logger = logging.getLogger(__name__)

def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    strategy: str = "f1",
    min_recall: float = 0.60,
) -> float:
    """
        Find the optimal decision threshold for fraud detection.

        The default 0.5 threshold is almost never correct for imbalanced
        fraud detection. With only 3.5% fraud the model is calibrated
        differently and we need to search for the best threshold.

        Strategies:
        - f1: maximize F1 score
        - recall_constrained: maximize precision subject to recall >= min_recall
    """
    curve_df = compute_precision_recall_curve(y_true, y_pred_proba)
    
    if len(curve_df) == 0:
        logger.warning("Empty precision recall curve. Returning default 0.5")
        return 0.5
    
    if strategy == "f1":
        best_idx = curve_df["f1"].idxmax()
        optimal_threshold = curve_df.loc[best_idx, "threshold"]
        best_f1 = curve_df.loc[best_idx, "f1"]
        best_precision = curve_df.loc[best_idx, "precision"]
        best_recall = curve_df.loc[best_idx, "recall"]
        
        logger.info(f"Optimal threshold (f1 strategy): {optimal_threshold:.4f}")
        logger.info(f"  F1: {best_f1:.4f}")
        logger.info(f"  Precision: {best_precision:.4f}")
        logger.info(f"  Recall: {best_recall:.4f}")
        
    elif strategy == "recall_constrained":
        valid = curve_df[curve_df["recall"] >= min_recall]
        
        if len(valid) == 0:
            logger.warning(
                f"No threshold achieves recall >= {min_recall}. "
                f"Falling back to f1 strategy."
            )
            return find_optimal_threshold(y_true, y_pred_proba, strategy="f1")
        
        # Among valid thresholds maximize precision
        best_idx = valid["precision"].idxmax()
        optimal_threshold = valid.loc[best_idx, "threshold"]
        best_f1 = valid.loc[best_idx, "f1"]
        best_precision = valid.loc[best_idx, "precision"]
        best_recall = valid.loc[best_idx, "recall"]
        
        logger.info(f" Optimal threshold (recall_constrained, min_recall={min_recall}): {optimal_threshold:.4f}")
        logger.info(f" F1: {best_f1:.4f}")
        logger.info(f" Precision: {best_precision:.4f}")
        logger.info(f" Recall: {best_recall:.4f}")
    
    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Choose 'f1' or 'recall_constrained'."
        )

    return float(optimal_threshold)

def get_threshold_analysis(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds_to_check: list = None,
) -> pd.DataFrame:
    """
        Compute metrics at several specific thresholds.
        Useful for understanding the precision-recall tradeoff
        at different operating points.

        Returns dataframe showing metrics at each threshold.
    """
    if thresholds_to_check is None:
        thresholds_to_check = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
    rows = []
    for threshold in thresholds_to_check:
        metrics = evaluate_model(y_true, y_pred_proba, threshold=threshold)
        rows.append(metrics)
    
    return pd.DataFrame(rows)
