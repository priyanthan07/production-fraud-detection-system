import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import logging
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import shap

logger = logging.getLogger(__name__)

MAX_SHAP_ROWS = 2000
BACKGROUND_SAMPLE_SIZE = 100
MAX_WATERFALL_PLOTS = 5
TOP_N_FEATURES = 20


def compute_shap_values(
    model,
    X: pd.DataFrame,
    model_name: str,
) -> tuple:
    """
    Compute SHAP values using TreeExplainer.

    TreeExplainer is exact (not approximate) for tree-based models
    and handles XGBoost, LightGBM, and CatBoost natively without
    any special casing.
    """
    logger.info(f"Computing SHAP values for {model_name} on {len(X)} rows...")

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X)

    # Older shap versions return [class_0_values, class_1_values].
    # Newer versions return a single array for binary classifiers.
    # Always extract the positive (fraud) class.

    if isinstance(raw, list):
        shap_values = raw[1]
    else:
        shap_values = raw

    logger.info(f"SHAP values shape: {shap_values.shape}")
    return explainer, shap_values


def plot_summary_beeswarm(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    model_name: str,
    output_path: str,
) -> str:
    """
    Generate a SHAP beeswarm summary plot.

    Each dot is one sample. The x-axis shows the SHAP value (impact
    on model output). Colour encodes the feature value (red = high,
    blue = low). Features are ranked by mean absolute SHAP value so
    the most important features appear at the top.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    shap.summary_plot(
        shap_values,
        X,
        plot_type="dot",
        max_display=TOP_N_FEATURES,
        show=False,
    )

    plt.title(
        f"{model_name} — SHAP Beeswarm (top {TOP_N_FEATURES} features)",
        pad=14,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved beeswarm plot: {output_path}")
    return output_path


def plot_summary_bar(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    model_name: str,
    output_path: str,
) -> str:
    """
    Generate a SHAP mean absolute value bar chart.

    Shows the average magnitude of each feature's SHAP value across
    all predictions. Simpler than beeswarm but easier for stakeholders
    who need a ranked feature importance list.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        max_display=TOP_N_FEATURES,
        show=False,
    )

    plt.title(
        f"{model_name} — SHAP Feature Importance (top {TOP_N_FEATURES})",
        pad=14,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved bar plot: {output_path}")
    return output_path


def plot_waterfall(
    explainer: shap.TreeExplainer,
    shap_values: np.ndarray,
    X: pd.DataFrame,
    row_index: int,
    model_name: str,
    label: str,
    output_path: str,
) -> str:
    """
    Generate a SHAP waterfall plot for a single prediction.

    Decomposes the model output for one transaction into contributions
    from each feature, starting from the expected value and showing
    how each feature pushes the prediction up (red) or down (blue).

    This is the primary tool for explaining individual fraud decisions
    to analysts or for debugging unexpected predictions.
    """
    # shap.Explanation wraps values, base value, and feature data into
    # the object expected by waterfall_plot (shap >= 0.40 API).
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        if len(base) > 1:
            base = base[1]
        else:
            base = base[0]
    base = float(base)

    explanation = shap.Explanation(
        values=shap_values[row_index],
        base_values=base,
        data=X.iloc[row_index].values,
        feature_names=list(X.columns),
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.waterfall_plot(explanation, max_display=15, show=False)

    amt = X.iloc[row_index].get("TransactionAmt", "?")
    try:
        title_amt = f"{float(amt):.2f}"
    except (TypeError, ValueError):
        title_amt = str(amt)

    plt.title(
        f"{model_name} — Waterfall ({label}, TransactionAmt={title_amt})",
        pad=14,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved waterfall plot: {output_path}")
    return output_path


def build_shap_feature_importance(
    shap_values: np.ndarray,
    feature_names: list,
) -> pd.DataFrame:
    """
    Build a DataFrame of mean absolute SHAP values per feature,
    sorted descending.

    Saved as a CSV artifact so downstream tools (dashboards, model
    cards) can consume feature importance without re-running SHAP.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)

    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    importance_df["rank"] = importance_df.index + 1
    return importance_df


def run_shap_analysis(
    model,
    model_name: str,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    """
    Run the full SHAP analysis pipeline for one model and log all
    artifacts to the active MLflow run.

    Steps:
    1. Sample validation set if too large (controls runtime)
    2. Compute SHAP values via TreeExplainer
    3. Save beeswarm summary plot
    4. Save bar chart summary plot
    5. Save waterfall plots for fraud and legitimate samples
    6. Save feature importance CSV
    7. Log all files as MLflow artifacts under shap/{model_name}/
    8. Log top-10 SHAP importance values as MLflow metrics

    Returns dict with paths to all generated artifacts and the
    importance DataFrame for downstream use.
    """
    logger.info(f"Running SHAP analysis for {model_name}...")

    # Cap at MAX_SHAP_ROWS. TreeExplainer is O(n * features * depth)
    # so 2000 rows on 383 features takes under a minute.
    if len(X_val) > MAX_SHAP_ROWS:
        logger.info(
            f"Validation set has {len(X_val)} rows. "
            f"Sampling {MAX_SHAP_ROWS} rows for SHAP to control runtime."
        )
        sample_idx = (
            pd.Series(range(len(X_val)))
            .sample(MAX_SHAP_ROWS, random_state=42)
            .sort_values()  # preserve time order within sample
            .values
        )
        X_shap = X_val.iloc[sample_idx].reset_index(drop=True)
        y_shap = y_val.iloc[sample_idx].reset_index(drop=True)
    else:
        X_shap = X_val.reset_index(drop=True)
        y_shap = y_val.reset_index(drop=True)

    explainer, shap_values = compute_shap_values(model, X_shap, model_name)

    artifacts = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # ----------------------------------------------------------------
        # Summary plots
        # ----------------------------------------------------------------
        beeswarm_path = str(tmp / f"{model_name}_shap_beeswarm.png")
        plot_summary_beeswarm(shap_values, X_shap, model_name, beeswarm_path)
        artifacts["beeswarm"] = beeswarm_path

        bar_path = str(tmp / f"{model_name}_shap_bar.png")
        plot_summary_bar(shap_values, X_shap, model_name, bar_path)
        artifacts["bar"] = bar_path

        # ----------------------------------------------------------------
        # Waterfall plots — sample of fraud and legitimate cases
        # ----------------------------------------------------------------
        fraud_indices = np.where(y_shap.values == 1)[0]
        legit_indices = np.where(y_shap.values == 0)[0]

        waterfall_paths = []

        for i, idx in enumerate(fraud_indices[:MAX_WATERFALL_PLOTS]):
            path = str(tmp / f"{model_name}_waterfall_fraud_{i}.png")
            plot_waterfall(
                explainer,
                shap_values,
                X_shap,
                row_index=int(idx),
                model_name=model_name,
                label=f"fraud_{i}",
                output_path=path,
            )
            waterfall_paths.append(path)

        for i, idx in enumerate(legit_indices[:MAX_WATERFALL_PLOTS]):
            path = str(tmp / f"{model_name}_waterfall_legit_{i}.png")
            plot_waterfall(
                explainer,
                shap_values,
                X_shap,
                row_index=int(idx),
                model_name=model_name,
                label=f"legit_{i}",
                output_path=path,
            )
            waterfall_paths.append(path)

        artifacts["waterfall_plots"] = waterfall_paths

        # ----------------------------------------------------------------
        # Feature importance CSV
        # ----------------------------------------------------------------
        importance_df = build_shap_feature_importance(shap_values, list(X_shap.columns))
        importance_path = str(tmp / f"{model_name}_shap_importance.csv")
        importance_df.to_csv(importance_path, index=False)
        artifacts["importance_csv"] = importance_path

        logger.info(
            f"Top 5 features by SHAP for {model_name}:\n"
            + importance_df.head(5).to_string(index=False)
        )

        # ----------------------------------------------------------------
        # Log everything to MLflow under shap/{model_name}/
        # ----------------------------------------------------------------
        artifact_subdir = f"shap/{model_name}"

        mlflow.log_artifact(beeswarm_path, artifact_path=artifact_subdir)
        mlflow.log_artifact(bar_path, artifact_path=artifact_subdir)
        mlflow.log_artifact(importance_path, artifact_path=artifact_subdir)

        for wp in waterfall_paths:
            mlflow.log_artifact(wp, artifact_path=f"{artifact_subdir}/waterfall")

        # Log top-10 as MLflow metrics so they are queryable from the UI
        for _, row in importance_df.head(10).iterrows():
            safe_name = row["feature"].replace(" ", "_")
            mlflow.log_metric(
                f"shap_importance_{safe_name}",
                round(float(row["mean_abs_shap"]), 6),
            )

        logger.info(f"SHAP artifacts logged to MLflow under '{artifact_subdir}'")

    artifacts["importance_df"] = importance_df
    return artifacts
