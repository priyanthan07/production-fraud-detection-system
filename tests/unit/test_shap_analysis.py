import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from unittest.mock import patch, MagicMock
from sklearn.ensemble import GradientBoostingClassifier

from src.explainability.shap_analysis import (
    compute_shap_values,
    build_shap_feature_importance,
    plot_summary_beeswarm,
    plot_summary_bar,
    plot_waterfall,
    run_shap_analysis,
    MAX_WATERFALL_PLOTS,
)


# ----------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------


def make_sample_data(n=200, n_features=10, fraud_rate=0.1):
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(n, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.binomial(1, fraud_rate, n), name="isFraud")
    return X, y


def make_trained_model(X, y):
    model = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def sample_model_and_data():
    X, y = make_sample_data()
    model = make_trained_model(X, y)
    return model, X, y


# ----------------------------------------------------------------
# compute_shap_values
# ----------------------------------------------------------------


def test_compute_shap_values_returns_tuple(sample_model_and_data):
    model, X, y = sample_model_and_data
    explainer, shap_values = compute_shap_values(model, X, "test_model")
    assert isinstance(shap_values, np.ndarray)


def test_shap_values_shape_matches_input(sample_model_and_data):
    model, X, y = sample_model_and_data
    _, shap_values = compute_shap_values(model, X, "test_model")
    assert shap_values.shape == X.shape


def test_shap_values_are_finite(sample_model_and_data):
    model, X, y = sample_model_and_data
    _, shap_values = compute_shap_values(model, X, "test_model")
    assert np.isfinite(shap_values).all()


def test_shap_values_not_all_zero(sample_model_and_data):
    model, X, y = sample_model_and_data
    _, shap_values = compute_shap_values(model, X, "test_model")
    assert np.abs(shap_values).sum() > 0


# ----------------------------------------------------------------
# build_shap_feature_importance
# ----------------------------------------------------------------


def test_importance_df_columns():
    shap_values = np.random.randn(100, 5)
    df = build_shap_feature_importance(shap_values, [f"f{i}" for i in range(5)])
    assert {"feature", "mean_abs_shap", "rank"}.issubset(df.columns)


def test_importance_df_sorted_descending():
    shap_values = np.random.randn(100, 5)
    df = build_shap_feature_importance(shap_values, [f"f{i}" for i in range(5)])
    assert df["mean_abs_shap"].is_monotonic_decreasing


def test_importance_df_row_count_matches_features():
    shap_values = np.random.randn(50, 8)
    df = build_shap_feature_importance(shap_values, [f"f{i}" for i in range(8)])
    assert len(df) == 8


def test_importance_df_rank_starts_at_one():
    shap_values = np.random.randn(50, 4)
    df = build_shap_feature_importance(shap_values, [f"f{i}" for i in range(4)])
    assert df["rank"].iloc[0] == 1


def test_importance_values_non_negative():
    shap_values = np.random.randn(50, 4)
    df = build_shap_feature_importance(shap_values, [f"f{i}" for i in range(4)])
    assert (df["mean_abs_shap"] >= 0).all()


# ----------------------------------------------------------------
# plot_summary_beeswarm
# ----------------------------------------------------------------


def test_beeswarm_creates_file(sample_model_and_data):
    model, X, y = sample_model_and_data
    _, shap_values = compute_shap_values(model, X, "test_model")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "beeswarm.png")
        result = plot_summary_beeswarm(shap_values, X, "test_model", path)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0


# ----------------------------------------------------------------
# plot_summary_bar
# ----------------------------------------------------------------


def test_bar_creates_file(sample_model_and_data):
    model, X, y = sample_model_and_data
    _, shap_values = compute_shap_values(model, X, "test_model")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bar.png")
        result = plot_summary_bar(shap_values, X, "test_model", path)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0


# ----------------------------------------------------------------
# plot_waterfall
# ----------------------------------------------------------------


def test_waterfall_creates_file(sample_model_and_data):
    model, X, y = sample_model_and_data
    explainer, shap_values = compute_shap_values(model, X, "test_model")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "waterfall.png")
        result = plot_waterfall(
            explainer,
            shap_values,
            X,
            row_index=0,
            model_name="test_model",
            label="fraud_0",
            output_path=path,
        )
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0


def test_waterfall_works_for_different_row_indices(sample_model_and_data):
    model, X, y = sample_model_and_data
    explainer, shap_values = compute_shap_values(model, X, "test_model")
    with tempfile.TemporaryDirectory() as tmpdir:
        for row_idx in [0, 1, len(X) - 1]:
            path = os.path.join(tmpdir, f"waterfall_{row_idx}.png")
            plot_waterfall(
                explainer,
                shap_values,
                X,
                row_index=row_idx,
                model_name="test_model",
                label=f"row_{row_idx}",
                output_path=path,
            )
            assert os.path.exists(path)


# ----------------------------------------------------------------
# run_shap_analysis (MLflow mocked)
# ----------------------------------------------------------------


@patch("src.explainability.shap_analysis.mlflow")
def test_run_shap_analysis_returns_expected_keys(mock_mlflow, sample_model_and_data):
    model, X, y = sample_model_and_data
    mock_mlflow.log_artifact = MagicMock()
    mock_mlflow.log_metric = MagicMock()
    result = run_shap_analysis(model, "test_model", X, y)
    assert {
        "beeswarm",
        "bar",
        "waterfall_plots",
        "importance_csv",
        "importance_df",
    }.issubset(result)


@patch("src.explainability.shap_analysis.mlflow")
def test_run_shap_analysis_calls_log_artifact(mock_mlflow, sample_model_and_data):
    model, X, y = sample_model_and_data
    mock_mlflow.log_artifact = MagicMock()
    mock_mlflow.log_metric = MagicMock()
    run_shap_analysis(model, "test_model", X, y)
    # beeswarm + bar + importance CSV = 3 mandatory calls minimum
    assert mock_mlflow.log_artifact.call_count >= 3


@patch("src.explainability.shap_analysis.mlflow")
def test_run_shap_analysis_logs_top10_metrics(mock_mlflow):
    mock_mlflow.log_artifact = MagicMock()
    mock_mlflow.log_metric = MagicMock()
    X, y = make_sample_data(n=50, n_features=10, fraud_rate=0.2)
    model = make_trained_model(X, y)
    run_shap_analysis(model, "test_model", X, y)
    metric_calls = [
        c for c in mock_mlflow.log_metric.call_args_list if "shap_importance" in str(c)
    ]
    assert len(metric_calls) == 10


@patch("src.explainability.shap_analysis.mlflow")
def test_importance_df_length_equals_n_features(mock_mlflow, sample_model_and_data):
    model, X, y = sample_model_and_data
    mock_mlflow.log_artifact = MagicMock()
    mock_mlflow.log_metric = MagicMock()
    result = run_shap_analysis(model, "test_model", X, y)
    assert len(result["importance_df"]) == X.shape[1]


@patch("src.explainability.shap_analysis.mlflow")
def test_large_val_set_is_sampled(mock_mlflow):
    mock_mlflow.log_artifact = MagicMock()
    mock_mlflow.log_metric = MagicMock()
    X, y = make_sample_data(n=3000, n_features=10, fraud_rate=0.1)
    model = make_trained_model(X, y)
    result = run_shap_analysis(model, "test_model", X, y)
    assert "importance_df" in result


@patch("src.explainability.shap_analysis.mlflow")
def test_waterfall_count_capped(mock_mlflow):
    mock_mlflow.log_artifact = MagicMock()
    mock_mlflow.log_metric = MagicMock()
    X, y = make_sample_data(n=200, n_features=10, fraud_rate=0.5)
    model = make_trained_model(X, y)
    result = run_shap_analysis(model, "test_model", X, y)
    assert len(result["waterfall_plots"]) <= MAX_WATERFALL_PLOTS * 2
