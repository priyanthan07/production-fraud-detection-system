import logging
import subprocess
import sys
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime


from src.monitoring.drift_detector import DriftDetector
from src.registry.model_manager import run_promotion_workflow as promote

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

# Path where retraining decisions are logged
RETRAINING_LOG_PATH = Path("data/drift_reports/retraining_log.csv")


def check_drift() -> dict:
    """
    Run drift detection and return the result.

    This is a thin wrapper around DriftDetector.run() that handles
    the case where drift detection fails (missing data, config errors).
    In production, a failed drift check should not trigger retraining —
    it should alert the on-call engineer.
    """

    try:
        detector = DriftDetector()
        result = detector.run()
        return result

    except FileNotFoundError as e:
        logger.error(f"Drift detection failed — missing file: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "skipped": True,
            "should_retrain": False,
            "reason": f"Drift detection failed: {e}",
            "error": str(e),
        }

    except Exception as e:
        logger.error(f"Drift detection failed unexpectedly: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "skipped": True,
            "should_retrain": False,
            "reason": f"Drift detection error: {e}",
            "error": str(e),
        }


def run_training_pipeline() -> dict:
    """
    Execute the full training pipeline as a subprocess.

    We run train.py as a separate process rather than importing
    and calling main() directly
    """
    logger.info("=" * 50)
    logger.info("Starting retraining pipeline...")
    logger.info("=" * 50)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.training.train"],
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout for training
        )

        if result.returncode == 0:
            logger.info("Training pipeline completed successfully.")
            return {
                "success": True,
                "stdout": result.stdout[-2000:] if result.stdout else "",
                "stderr": result.stderr[-500:] if result.stderr else "",
            }
        else:
            logger.error(
                f"Training pipeline failed with return code {result.returncode}"
            )
            logger.error(f"stderr: {result.stderr[-1000:]}")
            return {
                "success": False,
                "return_code": result.returncode,
                "stdout": result.stdout[-2000:] if result.stdout else "",
                "stderr": result.stderr[-1000:] if result.stderr else "",
            }

    except subprocess.TimeoutExpired:
        logger.error("Training pipeline timed out after 2 hours.")
        return {
            "success": False,
            "error": "Training timed out after 7200 seconds",
        }

    except Exception as e:
        logger.error(f"Failed to start training pipeline: {e}")
        return {
            "success": False,
            "error": str(e),
        }


def run_promotion_workflow() -> dict:
    """
    Run the model promotion workflow after training.
    """

    logger.info("Running model promotion workflow...")

    try:
        result = promote(auto_select=True)

        if result["promoted"]:
            logger.info(
                f"Model version {result['version']} promoted to Production. Threshold: {result['optimal_threshold']}"
            )
        else:
            logger.warning(
                f"Model version {result['version']} failed quality gates. Not promoted."
            )

        return {
            "success": True,
            "promoted": result["promoted"],
            "version": result["version"],
            "gates_passed": result["gates_passed"],
            "gate_report": result["gate_report"],
        }

    except Exception as e:
        logger.error(f"Promotion workflow failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


def log_retraining_event(
    drift_result: dict,
    training_result: dict = None,
    promotion_result: dict = None,
    forced: bool = False,
) -> None:
    """
    Append a record to the retraining log CSV.
    """
    RETRAINING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now().isoformat(),
        "forced": forced,
        "drift_detected": drift_result.get("should_retrain", False),
        "drift_reason": drift_result.get("reason", ""),
        "drift_fraction": (drift_result.get("summary", {}).get("drift_fraction", None)),
        "training_triggered": training_result is not None,
        "training_success": (
            training_result.get("success", False) if training_result else False
        ),
        "promotion_triggered": promotion_result is not None,
        "promoted": (
            promotion_result.get("promoted", False) if promotion_result else False
        ),
        "new_model_version": (
            promotion_result.get("version", None) if promotion_result else None
        ),
    }

    row_df = pd.DataFrame([row])

    if RETRAINING_LOG_PATH.exists():
        existing = pd.read_csv(RETRAINING_LOG_PATH)
        row_df = pd.concat([existing, row_df], ignore_index=True)

    row_df.to_csv(RETRAINING_LOG_PATH, index=False)
    logger.info(f"Retraining event logged to {RETRAINING_LOG_PATH}")


def run(force: bool = False) -> dict:
    """
    Main entry point for the retraining trigger.

    Steps:
    1. Run drift detection (unless force=True)
    2. If drift detected or forced: run training pipeline
    3. If training succeeds: run promotion workflow
    4. Log everything
    """

    logger.info("=" * 60)
    logger.info("Retraining Trigger")
    logger.info(f"  Mode: {'FORCED' if force else 'DRIFT-TRIGGERED'}")
    logger.info(f"  Time: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Step 1: Check drift
    if force:
        drift_result = {
            "timestamp": datetime.now().isoformat(),
            "should_retrain": True,
            "reason": "Forced retraining requested",
            "skipped": True,
        }
        logger.info("Forced retraining — skipping drift detection.")
    else:
        drift_result = check_drift()

    # Step 2: Decide whether to retrain
    if not drift_result["should_retrain"]:
        logger.info(f"No retraining needed. Reason: {drift_result['reason']}")

        log_retraining_event(drift_result, forced=force)
        return {
            "retrained": False,
            "drift_result": drift_result,
        }

    # Step 3: Run training
    logger.info(f"Retraining triggered. Reason: {drift_result['reason']}")
    training_result = run_training_pipeline()

    if not training_result["success"]:
        logger.error("Training failed. Aborting promotion.")
        log_retraining_event(drift_result, training_result, forced=force)
        return {
            "retrained": False,
            "drift_result": drift_result,
            "training_result": training_result,
            "error": "Training pipeline failed",
        }

    # Step 4: Run promotion
    promotion_result = run_promotion_workflow()

    # Step 5: Log everything
    log_retraining_event(drift_result, training_result, promotion_result, forced=force)

    return {
        "retrained": True,
        "drift_result": drift_result,
        "training_result": {
            "success": training_result["success"],
        },
        "promotion_result": promotion_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check drift and retrain if needed")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining regardless of drift detection",
    )
    args = parser.parse_args()

    result = run(force=args.force)

    print("\n" + "=" * 50)
    print("Retraining Trigger Result")
    print("=" * 50)
    print(f"  Retrained: {result['retrained']}")

    if result.get("promotion_result"):
        pr = result["promotion_result"]
        print(f"  Promoted:  {pr.get('promoted', False)}")
        print(f"  Version:   {pr.get('version', 'N/A')}")
    else:
        reason = result.get("drift_result", {}).get("reason", "")
        print(f"  Reason:    {reason}")
