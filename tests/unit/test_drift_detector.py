"""
Tests for drift detection.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.monitoring.drift_detector import compute_psi, DriftDetector


# ================================================================
# Tests for compute_psi
# ================================================================

class TestComputePSI:
    """Tests for the standalone PSI computation function."""

    def test_identical_distributions_psi_near_zero(self):
        """When expected and actual are the same, PSI should be ~0."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 10000))
        psi = compute_psi(data, data)
        assert psi < 0.01, f"PSI for identical data should be ~0, got {psi}"

    def test_similar_distributions_low_psi(self):
        """Slightly different distributions should have low PSI."""
        np.random.seed(42)
        expected = pd.Series(np.random.normal(0, 1, 10000))
        actual = pd.Series(np.random.normal(0.05, 1, 10000))
        psi = compute_psi(expected, actual)
        assert psi < 0.10, f"Similar distributions should have PSI < 0.10, got {psi}"

    def test_different_distributions_high_psi(self):
        """Very different distributions should have high PSI."""
        np.random.seed(42)
        expected = pd.Series(np.random.normal(0, 1, 10000))
        actual = pd.Series(np.random.normal(3, 1, 10000))
        psi = compute_psi(expected, actual)
        assert psi > 0.25, f"Very different distributions should have PSI > 0.25, got {psi}"

    def test_psi_is_non_negative(self):
        """PSI should always be >= 0."""
        np.random.seed(42)
        expected = pd.Series(np.random.uniform(0, 100, 5000))
        actual = pd.Series(np.random.uniform(10, 90, 5000))
        psi = compute_psi(expected, actual)
        assert psi >= 0, f"PSI should be non-negative, got {psi}"

    def test_psi_handles_nan_values(self):
        """PSI should work when data contains NaN values."""
        np.random.seed(42)
        expected = pd.Series([1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10] * 100)
        actual = pd.Series([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10] * 100)
        psi = compute_psi(expected, actual)
        assert isinstance(psi, float)
        assert psi >= 0

    def test_psi_with_too_few_values_returns_zero(self):
        """PSI should return 0.0 when there are too few values."""
        expected = pd.Series([1, 2, 3])
        actual = pd.Series([4, 5, 6])
        psi = compute_psi(expected, actual)
        assert psi == 0.0

    def test_psi_with_all_nan_returns_zero(self):
        """PSI should return 0.0 when all values are NaN."""
        expected = pd.Series([np.nan] * 100)
        actual = pd.Series([np.nan] * 100)
        psi = compute_psi(expected, actual)
        assert psi == 0.0

    def test_psi_with_constant_values(self):
        """PSI should handle constant-value features gracefully."""
        expected = pd.Series([5.0] * 1000)
        actual = pd.Series([5.0] * 1000)
        psi = compute_psi(expected, actual)
        assert psi == 0.0

    def test_psi_moderate_shift(self):
        """A moderate distribution shift should produce moderate PSI."""
        np.random.seed(42)
        expected = pd.Series(np.random.normal(0, 1, 10000))
        actual = pd.Series(np.random.normal(0.5, 1.2, 10000))
        psi = compute_psi(expected, actual)
        assert 0.05 < psi < 0.50, f"Moderate shift PSI should be in 0.05-0.50, got {psi}"


# ================================================================
# Tests for DriftDetector
# ================================================================

class TestDriftDetector:
    """Tests for the DriftDetector class."""

    @pytest.fixture
    def mock_config(self):
        return {
            "baseline_data_path": "data/processed/train_features.parquet",
            "production_data_path": "data/production/recent_predictions.parquet",
            "drift_report_path": "data/drift_reports",
            "drift_history_path": "data/drift_reports/drift_history.csv",
            "psi_warning_threshold": 0.10,
            "psi_critical_threshold": 0.25,
            "drift_feature_fraction": 0.20,
            "target_column": "isFraud",
            "target_drift_threshold": 0.05,
            "monitored_features": ["feature_a", "feature_b", "feature_c"],
            "min_sample_size": 100,
            "schedule_interval": "0 6 * * *",
        }

    @pytest.fixture
    def sample_baseline(self):
        np.random.seed(42)
        return pd.DataFrame({
            "feature_a": np.random.normal(0, 1, 5000),
            "feature_b": np.random.uniform(0, 100, 5000),
            "feature_c": np.random.exponential(2, 5000),
            "isFraud": np.random.binomial(1, 0.035, 5000),
        })

    @pytest.fixture
    def sample_production_no_drift(self):
        np.random.seed(43)
        return pd.DataFrame({
            "feature_a": np.random.normal(0, 1, 1000),
            "feature_b": np.random.uniform(0, 100, 1000),
            "feature_c": np.random.exponential(2, 1000),
            "isFraud": np.random.binomial(1, 0.035, 1000),
        })

    @pytest.fixture
    def sample_production_with_drift(self):
        np.random.seed(44)
        return pd.DataFrame({
            "feature_a": np.random.normal(3, 2, 1000),
            "feature_b": np.random.uniform(50, 200, 1000),
            "feature_c": np.random.exponential(10, 1000),
            "isFraud": np.random.binomial(1, 0.15, 1000),
        })

    @patch("src.monitoring.drift_detector.load_drift_config")
    def test_no_drift_detected(
        self, mock_load_config, mock_config,
        sample_baseline, sample_production_no_drift, tmp_path
    ):
        mock_config["drift_report_path"] = str(tmp_path)
        mock_config["drift_history_path"] = str(tmp_path / "history.csv")
        mock_load_config.return_value = mock_config

        detector = DriftDetector()
        detector.baseline_df = sample_baseline
        detector.production_df = sample_production_no_drift

        result = detector.run()

        assert result["should_retrain"] is False
        assert result["summary"]["critical"] == 0

    @patch("src.monitoring.drift_detector.load_drift_config")
    def test_drift_detected(
        self, mock_load_config, mock_config,
        sample_baseline, sample_production_with_drift, tmp_path
    ):
        mock_config["drift_report_path"] = str(tmp_path)
        mock_config["drift_history_path"] = str(tmp_path / "history.csv")
        mock_load_config.return_value = mock_config

        detector = DriftDetector()
        detector.baseline_df = sample_baseline
        detector.production_df = sample_production_with_drift

        result = detector.run()

        assert result["should_retrain"] is True
        assert result["summary"]["critical"] > 0

    @patch("src.monitoring.drift_detector.load_drift_config")
    def test_insufficient_samples(
        self, mock_load_config, mock_config, sample_baseline, tmp_path
    ):
        mock_config["drift_report_path"] = str(tmp_path)
        mock_config["drift_history_path"] = str(tmp_path / "history.csv")
        mock_config["min_sample_size"] = 500
        mock_load_config.return_value = mock_config

        small_production = pd.DataFrame({
            "feature_a": [1, 2, 3],
            "feature_b": [4, 5, 6],
            "feature_c": [7, 8, 9],
        })

        detector = DriftDetector()
        detector.baseline_df = sample_baseline
        detector.production_df = small_production

        result = detector.run()

        assert result["skipped"] is True
        assert result["should_retrain"] is False

    @patch("src.monitoring.drift_detector.load_drift_config")
    def test_target_drift_triggers_retrain(
        self, mock_load_config, mock_config,
        sample_baseline, tmp_path
    ):
        mock_config["drift_report_path"] = str(tmp_path)
        mock_config["drift_history_path"] = str(tmp_path / "history.csv")
        mock_load_config.return_value = mock_config

        np.random.seed(45)
        production_target_drift = pd.DataFrame({
            "feature_a": np.random.normal(0, 1, 1000),
            "feature_b": np.random.uniform(0, 100, 1000),
            "feature_c": np.random.exponential(2, 1000),
            "isFraud": np.random.binomial(1, 0.15, 1000),
        })

        detector = DriftDetector()
        detector.baseline_df = sample_baseline
        detector.production_df = production_target_drift

        result = detector.run()

        assert result["target_drift"]["drifted"] is True

    @patch("src.monitoring.drift_detector.load_drift_config")
    def test_report_saved(
        self, mock_load_config, mock_config,
        sample_baseline, sample_production_no_drift, tmp_path
    ):
        mock_config["drift_report_path"] = str(tmp_path)
        mock_config["drift_history_path"] = str(tmp_path / "history.csv")
        mock_load_config.return_value = mock_config

        detector = DriftDetector()
        detector.baseline_df = sample_baseline
        detector.production_df = sample_production_no_drift

        detector.run()

        # Check report was saved
        reports = list(tmp_path.glob("drift_report_*.json"))
        assert len(reports) == 1

        # Check history was saved
        assert (tmp_path / "history.csv").exists()
        