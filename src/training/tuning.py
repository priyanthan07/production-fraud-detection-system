import numpy as np
import pandas as pd
import logging
import optuna
from sklearn.metrics import average_precision_score

from src.training.models.xgboost_model import train_xgboost
from src.training.models.lightgbm_model import train_lightgbm
from src.training.models.catboost_model import train_catboost

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose output
optuna.logging.set_verbosity(optuna.logging.WARNING)

def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 30,
) -> dict:
    """
        Use Optuna to find the best XGBoost hyperparameters.

        Optuna uses Bayesian optimization to intelligently search
        the hyperparameter space. Unlike grid search which tries
        every combination, Optuna learns from previous trials and
        focuses on promising regions of the search space.

        We optimize for AUC-PR because it is the most informative
        metric for imbalanced fraud detection.
    """
    
    def objective(trial):
        params = {
            "n_estimators" : trial.suggest_int("n_estimators", 100, 1000),
            "max_depth" : trial.suggest_int("max_depth", 3, 9),
            "learning_rate" : trial.suggest_int("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
        
        try:
            model = train_xgboost(X_train, y_train, X_val, y_val, params)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc_pr = average_precision_score(y_val, y_pred_proba)
            return auc_pr
        
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"XGBoost best AUC-PR: {study.best_value:.4f}")
    logger.info(f"XGBoost best params: {study.best_params}")

    return study.best_params

def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 30,
) -> dict:
    """
        Use Optuna to find the best LightGBM hyperparameters.
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }

        try:
            model = train_lightgbm(X_train, y_train, X_val, y_val, params)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc_pr = average_precision_score(y_val, y_pred_proba)
            return auc_pr
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"LightGBM best AUC-PR: {study.best_value:.4f}")
    logger.info(f"LightGBM best params: {study.best_params}")

    return study.best_params

def tune_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 30,
) -> dict:
    """
    Use Optuna to find the best CatBoost hyperparameters.
    """
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "depth": trial.suggest_int("depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.0, 10.0),
            "verbose": 0,
            "early_stopping_rounds": 50,
        }

        try:
            model = train_catboost(X_train, y_train, X_val, y_val, params)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc_pr = average_precision_score(y_val, y_pred_proba)
            return auc_pr
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"CatBoost best AUC-PR: {study.best_value:.4f}")
    logger.info(f"CatBoost best params: {study.best_params}")

    return study.best_params
