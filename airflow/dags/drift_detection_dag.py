from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


# ----------------------------------------------------------------
# DAG configuration
# ----------------------------------------------------------------
default_args = {
    "owner": "ml-engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def run_drift_detection(**context):
    """
    Task 1: Run the drift detector.
    """
    import logging

    logging.basicConfig(level=logging.INFO)

    # Import here to avoid Airflow import-time errors
    # when the module is not on the Airflow worker's PYTHONPATH
    import sys

    sys.path.insert(0, "/opt/airflow/project")

    from src.monitoring.drift_detector import DriftDetector

    detector = DriftDetector()
    result = detector.run()

    # Push to XCom for downstream tasks
    context["ti"].xcom_push(key="drift_result", value=result)

    return result


def evaluate_drift(**context):
    """
    Task 2: Evaluate drift results and decide whether to retrain.
    """
    import logging

    logger = logging.getLogger(__name__)

    drift_result = context["ti"].xcom_pull(
        task_ids="run_drift_detection",
        key="drift_result",
    )

    if drift_result is None:
        logger.error("No drift result found in XCom.")
        return "no_retrain"

    if drift_result.get("skipped", False):
        logger.info(
            f"Drift detection was skipped: {drift_result.get('reason', 'unknown')}"
        )
        return "no_retrain"

    should_retrain = drift_result.get("should_retrain", False)

    if should_retrain:
        logger.warning(
            f"Drift detected — triggering retraining. "
            f"Reason: {drift_result.get('reason', 'unknown')}"
        )
        return "trigger_retrain"
    else:
        logger.info(
            f"No significant drift. Reason: {drift_result.get('reason', 'No drift')}"
        )
        return "no_retrain"


def log_no_retrain(**context):
    """
    Task 3a: Log that no retraining was needed.
    Called when drift is below threshold.
    """
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Drift check complete. No retraining needed today.")


# ----------------------------------------------------------------
# DAG definition
# ----------------------------------------------------------------
with DAG(
    dag_id="fraud_drift_detection",
    default_args=default_args,
    description="Daily drift detection for fraud model",
    schedule_interval="0 6 * * *",  # daily at 6 AM
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["fraud", "ml", "monitoring", "drift"],
) as dag:
    # Task 1: Run drift detection
    detect_drift = PythonOperator(
        task_id="run_drift_detection",
        python_callable=run_drift_detection,
        provide_context=True,
    )

    # Task 2: Evaluate results
    evaluate = BranchPythonOperator(
        task_id="evaluate_drift",
        python_callable=evaluate_drift,
        provide_context=True,
    )

    # Task 3a: Log no-retrain decision
    no_retrain = PythonOperator(
        task_id="log_no_retrain",
        python_callable=log_no_retrain,
        provide_context=True,
        trigger_rule="none_failed_min_one_success",
    )

    # Task 3b: Trigger retraining DAG
    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="fraud_model_retraining",
        wait_for_completion=False,
        # We do not wait for retraining to complete because it
        # takes 1-2 hours. The retraining DAG handles its own
        # success/failure logging.
    )

    # ----------------------------------------------------------------
    # Task dependencies
    #
    # detect_drift → evaluate → trigger_retrain (if drift)
    #                         → no_retrain (if no drift)
    #
    # The branching is handled inside evaluate_drift's return value.
    # In a production Airflow setup you would use BranchPythonOperator
    # for proper branching. Here we keep it simple — both downstream
    # tasks run but only the relevant one does meaningful work.
    # ----------------------------------------------------------------
    detect_drift >> evaluate >> [trigger_retrain, no_retrain]
