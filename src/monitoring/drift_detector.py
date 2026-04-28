# Compares production data against the training baseline to detect data drift using PSI (Population Stability Index).

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Default config path
CONFIG_PATH = Path("configs/drift_config.yaml")


def load_drift_config(config_path: Path = CONFIG_PATH) -> dict:
    """
    Load drift detection configuration from YAML.
    """

    if not config_path.exists():
        raise FileNotFoundError(
            f"Drift config not found at {config_path}. Create it before running drift detection."
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info(
        f"Loaded drift config: monitoring {len(config['monitored_features'])} features, "
        f"PSI warning={config['psi_warning_threshold']}, "
        f"PSI critical={config['psi_critical_threshold']}"
    )
    return config


def compute_psi(expected: pd.Series, actual: pd.Series, n_bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) between two distributions.
    PSI measures how much the distribution of a feature has shifted
    between the training data (expected) and production data (actual).
    """

    expected_clean = expected.dropna()
    actual_clean = actual.dropna()

    if len(expected_clean) < 10 or len(actual_clean) < 10:
        logger.warning(
            f"Too few non-null values for PSI computation "
            f"(expected: {len(expected_clean)}, actual: {len(actual_clean)}). "
            f"Returning 0.0."
        )
        return 0.0

    # Create bins from expected distribution Using quantile bins ensures each
    # bin has roughly equal count in the training data

    try:
        bin_edges = np.percentile(
            expected_clean,
            np.linspace(0, 100, n_bins + 1),
        )

        # Remove duplicate edges (happens when many values are identical)
        bin_edges = np.unique(bin_edges)

        # Feature has very low cardinality — PSI is not meaningful
        if len(bin_edges) < 3:
            return 0.0

    except Exception as e:
        logger.warning(f"Could not compute bin edges: {e}")
        return 0.0

    # Count values in each bin for both distributions
    expected_counts = np.histogram(expected_clean, bins=bin_edges)[0]
    actual_counts = np.histogram(actual_clean, bins=bin_edges)[0]

    # Convert to proportions
    expected_pct = expected_counts / expected_counts.sum()
    actual_pct = actual_counts / actual_counts.sum()

    # Add epsilon to avoid log(0) and division by zero
    epsilon = 1e-4
    expected_pct = np.maximum(expected_pct, epsilon)
    actual_pct = np.maximum(actual_pct, epsilon)

    # PSI formula
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    return float(psi)


class DriftDetector:
    """
    Main drift detection class. Loads baseline (training) data and production data, computes
    PSI for each monitored feature, and determines whether the aggregate drift level warrants retraining.
    """

    def __init__(self, config_path: Path = CONFIG_PATH):
        self.config = load_drift_config(config_path)
        self.baseline_df = None
        self.production_df = None

    def load_baseline(self, path: str = None) -> pd.DataFrame:
        """
        Load the training data baseline.

        This is the data the model was trained on. We compute
        the expected distribution of each feature from this data.
        """

        path = path or self.config["baseline_data_path"]
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Baseline data not found at {path}. Run training first to generate processed features."
            )

        self.baseline_df = pd.read_parquet(path)
        logger.info(
            f"Loaded baseline data: {len(self.baseline_df)} rows, {len(self.baseline_df.columns)} columns"
        )
        return self.baseline_df

    def load_production_data(self, path: str = None) -> pd.DataFrame:
        """
        Load recent production data for drift comparison.

        In a real system this would query a feature store or
        prediction log database. For this project we read from
        a parquet file that the inference server writes to.
        """

        path = path or self.config["production_data_path"]
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Production data not found at {path}. The inference server needs to log predictions first."
            )

        self.production_df = pd.read_parquet(path)
        logger.info(
            f"Loaded production data: {len(self.production_df)} rows, {len(self.production_df.columns)} columns"
        )
        return self.production_df

    def compute_feature_drift(self) -> dict:
        """
        Compute PSI for each monitored feature.
        """

        if self.baseline_df is None:
            self.load_baseline()
        if self.production_df is None:
            self.load_production_data()

        results = {}
        monitored = self.config["monitored_features"]
        warning_threshold = self.config["psi_warning_threshold"]
        critical_threshold = self.config["psi_critical_threshold"]

        for feature in monitored:
            if feature not in self.baseline_df.columns:
                logger.warning(f"Feature '{feature}' not found in baseline data. Skipping.")
                continue

            if feature not in self.production_df.columns:
                logger.warning(f"Feature '{feature}' not found in production data. Skipping.")
                continue

            psi = compute_psi(
                expected=self.baseline_df[feature],
                actual=self.production_df[feature],
            )

            if psi >= critical_threshold:
                status = "CRITICAL"
            elif psi >= warning_threshold:
                status = "WARNING"
            else:
                status = "OK"

            results[feature] = {
                "psi": round(psi, 4),
                "status": status,
            }

            if status != "OK":
                logger.warning(f"Drift detected in '{feature}': PSI={psi:.4f} ({status})")

        return results

    def compute_target_drift(self) -> dict:
        """
        Check if the fraud rate itself has changed.

        If the baseline fraud rate was 3.5% and production data
        shows 7%, that is a strong signal that something has changed
        in the underlying fraud patterns.
        """

        target_col = self.config["target_column"]
        threshold = self.config["target_drift_threshold"]

        result = {
            "baseline_fraud_rate": None,
            "production_fraud_rate": None,
            "absolute_change": None,
            "drifted": False,
        }

        if target_col in self.baseline_df.columns:
            result["baseline_fraud_rate"] = round(float(self.baseline_df[target_col].mean()), 4)

        if target_col in self.production_df.columns:
            result["production_fraud_rate"] = round(float(self.production_df[target_col].mean()), 4)

            if result["baseline_fraud_rate"] is not None:
                change = abs(result["production_fraud_rate"] - result["baseline_fraud_rate"])

                result["absolute_change"] = round(change, 4)
                result["drifted"] = change > threshold

                if result["drifted"]:
                    logger.warning(
                        f"Target drift detected: fraud rate changed from {result['baseline_fraud_rate']:.4f} to "
                        f"{result['production_fraud_rate']:.4f} (change: {change:.4f}, threshold: {threshold})"
                    )

        else:
            logger.info(
                "Target column not in production data — target drift check skipped (expected for unlabeled data)."
            )

        return result

    def run(self) -> dict:
        """
        Run the full drift detection pipeline.

        Steps:
        1. Load baseline and production data
        2. Check minimum sample size
        3. Compute per-feature PSI
        4. Compute target drift (if labels available)
        5. Determine whether to retrain
        """

        logger.info("=" * 50)
        logger.info("Starting drift detection run...")
        logger.info("=" * 50)

        # Load data
        if self.baseline_df is None:
            self.load_baseline()
        if self.production_df is None:
            self.load_production_data()

        # Check minimum sample size
        min_samples = self.config["min_sample_size"]

        if len(self.production_df) < min_samples:
            logger.warning(
                f"Production data has only {len(self.production_df)} rows (minimum: {min_samples}). Skipping drift detection."
            )
            return {
                "timestamp": datetime.now().isoformat(),
                "skipped": True,
                "reason": f"Insufficient production data ({len(self.production_df)} < {min_samples})",
                "should_retrain": False,
            }

        # Compute feature drift
        feature_drift = self.compute_feature_drift()

        # Compute target drift
        target_drift = self.compute_target_drift()

        # Aggregate results
        total_features = len(feature_drift)
        critical_features = sum(1 for r in feature_drift.values() if r["status"] == "CRITICAL")

        warning_features = sum(1 for r in feature_drift.values() if r["status"] == "WARNING")
        ok_features = total_features - critical_features - warning_features

        drift_fraction = critical_features / total_features if total_features > 0 else 0.0

        # Decision: should we retrain?
        fraction_threshold = self.config["drift_feature_fraction"]
        should_retrain = drift_fraction >= fraction_threshold or target_drift.get("drifted", False)

        # Build reason string
        reasons = []
        if drift_fraction >= fraction_threshold:
            reasons.append(
                f"{critical_features}/{total_features} features  ({drift_fraction:.0%}) exceeded critical PSI threshold"
            )

        if target_drift.get("drifted", False):
            reasons.append(f"Target fraud rate shifted by {target_drift['absolute_change']:.4f}")

        reason = " AND ".join(reasons) if reasons else "No significant drift detected"

        # Log summary
        logger.info("\nDrift Detection Summary:")
        logger.info(f"  Features monitored: {total_features}")
        logger.info(f"  OK:       {ok_features}")
        logger.info(f"  WARNING:  {warning_features}")
        logger.info(f"  CRITICAL: {critical_features}")
        logger.info(f"  Drift fraction: {drift_fraction:.2%}")
        logger.info(f"  Should retrain: {should_retrain}")
        logger.info(f"  Reason: {reason}")

        result = {
            "timestamp": datetime.now().isoformat(),
            "skipped": False,
            "feature_drift": feature_drift,
            "target_drift": target_drift,
            "summary": {
                "total_features": total_features,
                "ok": ok_features,
                "warning": warning_features,
                "critical": critical_features,
                "drift_fraction": round(drift_fraction, 4),
                "baseline_rows": len(self.baseline_df),
                "production_rows": len(self.production_df),
            },
            "should_retrain": should_retrain,
            "reason": reason,
        }

        # Save report
        self._save_report(result)

        return result

    def _save_report(self, result: dict) -> None:
        """
        Save the drift report to disk for auditing.
        """

        report_dir = Path(self.config["drift_report_path"])
        report_dir.mkdir(parents=True, exist_ok=True)

        # Save full report as JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"drift_report_{timestamp}.json"

        with open(report_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"Drift report saved to {report_path}")

        # Append summary to history CSV for trend tracking
        history_path = Path(self.config["drift_history_path"])

        history_row = {
            "timestamp": result["timestamp"],
            "total_features": result["summary"]["total_features"],
            "ok": result["summary"]["ok"],
            "warning": result["summary"]["warning"],
            "critical": result["summary"]["critical"],
            "drift_fraction": result["summary"]["drift_fraction"],
            "should_retrain": result["should_retrain"],
            "reason": result["reason"],
        }

        history_df = pd.DataFrame([history_row])

        if history_path.exists():
            existing = pd.read_csv(history_path)
            history_df = pd.concat([existing, history_df], ignore_index=True)

        history_df.to_csv(history_path, index=False)
        logger.info(f"Drift history appended to {history_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    detector = DriftDetector()
    result = detector.run()
    print(f"\nShould retrain: {result['should_retrain']}")
    print(f"Reason: {result['reason']}")
