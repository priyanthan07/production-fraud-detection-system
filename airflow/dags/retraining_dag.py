from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator


# ----------------------------------------------------------------
# DAG configuration
# ----------------------------------------------------------------
default_args = {
    "owner": "ml-engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    # No retries for training — if it fails we want to investigate
    # the failure manually, not blindly retry. Training is expensive
    # and a retry with the same data will likely fail the same way.
    "retry_delay": timedelta(minutes=10),
}


def validate_data(**context):
    """
    Task 1: Load and validate the raw data.
    """
    import logging
    import sys
    from pathlib import Path
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    sys.path.insert(0, "/opt/airflow/project")

    from src.ingestion.loader import load_raw_data
    from src.ingestion.validator import validate_raw_data
    
    project_root = Path("/opt/airflow/project")
    raw_data_path = str(project_root / "data/raw")

    logger.info("Loading raw data...")
    raw_df = load_raw_data("data/raw")

    logger.info("Validating raw data...")
    validate_raw_data(raw_df)

    logger.info(
        f"Data validation passed. {len(raw_df)} rows, "
        f"{len(raw_df.columns)} columns."
    )

    # Push row count to XCom for logging
    context["ti"].xcom_push(key="row_count", value=len(raw_df))


def run_training(**context):
    """
    Task 2: Run the full training pipeline.
    """
    import logging
    import subprocess
    import sys
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting training pipeline...")

    result = subprocess.run(
        [sys.executable, "-m", "src.training.train"],
        capture_output=True,
        text=True,
        timeout=7200,
        cwd="/opt/airflow/project",
    )

    if result.returncode != 0:
        logger.error(f"Training failed:\n{result.stderr[-2000:]}")
        raise RuntimeError(
            f"Training pipeline failed with code {result.returncode}"
        )

    logger.info("Training pipeline completed successfully.")

    # Extract best model info from stdout
    context["ti"].xcom_push(
        key="training_output",
        value=result.stdout[-3000:],
    )


def promote_model(**context):
    """
    Task 3: Run the model promotion workflow.

    """
    import logging
    import sys
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    sys.path.insert(0, "/opt/airflow/project")

    from src.registry.model_manager import run_promotion_workflow

    result = run_promotion_workflow(auto_select=True)

    if result["promoted"]:
        logger.info(
            f"Model version {result['version']} promoted to Production!"
        )
    else:
        logger.warning(
            f"Model version {result['version']} failed quality gates. "
            f"Current Production model remains unchanged."
        )

    context["ti"].xcom_push(key="promotion_result", value={
        "promoted": result["promoted"],
        "version": result["version"],
        "gates_passed": result["gates_passed"],
    })


def save_new_baseline(**context):
    """
    Task 4: Update the drift baseline after successful retraining.
    """
    import logging
    import shutil
    from pathlib import Path
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    promotion_result = context["ti"].xcom_pull(
        task_ids="promote_model",
        key="promotion_result",
    )

    if not promotion_result or not promotion_result.get("promoted"):
        logger.info("Model was not promoted — keeping existing baseline.")
        return

    # Use absolute path based on project root mounted in container
    project_root = Path("/opt/airflow/project")
    source = project_root / "data/processed/train_features.parquet"
    baseline = project_root / "data/processed/drift_baseline.parquet"

    if source.exists():
        shutil.copy2(source, baseline)
        logger.info(f"Updated drift baseline from {source} to {baseline}")
    else:
        logger.warning(f"Source file {source} not found. Baseline not updated.")

def log_retraining_complete(**context):
    """
    Task 5: Log the retraining completion for auditing.
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    promotion_result = context["ti"].xcom_pull(
        task_ids="promote_model",
        key="promotion_result",
    )

    logger.info("=" * 50)
    logger.info("Retraining Pipeline Complete")
    logger.info("=" * 50)

    if promotion_result:
        logger.info(f"  Promoted: {promotion_result.get('promoted')}")
        logger.info(f"  Version:  {promotion_result.get('version')}")
        logger.info(f"  Gates:    {promotion_result.get('gates_passed')}")
    else:
        logger.info("  No promotion result available.")


# ----------------------------------------------------------------
# DAG definition
# ----------------------------------------------------------------
with DAG(
    dag_id="fraud_model_retraining",
    default_args=default_args,
    description="Full model retraining pipeline for fraud detection",
    schedule_interval=None,
    # schedule_interval=None means this DAG is only triggered
    # manually or by the drift detection DAG. It does not run
    # on a fixed schedule.
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    # max_active_runs=1 prevents two retraining runs from
    # happening simultaneously. Training is resource-intensive
    # and concurrent runs would compete for memory and CPU.
    tags=["fraud", "ml", "retraining"],
) as dag:

    # Task 1: Validate data
    validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
        provide_context=True,
    )

    # Task 2: Train models
    train = PythonOperator(
        task_id="run_training",
        python_callable=run_training,
        provide_context=True,
        execution_timeout=timedelta(hours=3),
    )

    # Task 3: Promote best model
    promote = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model,
        provide_context=True,
    )

    # Task 4: Update drift baseline
    update_baseline = PythonOperator(
        task_id="save_new_baseline",
        python_callable=save_new_baseline,
        provide_context=True,
    )

    # Task 5: Log completion
    log_complete = PythonOperator(
        task_id="log_retraining_complete",
        python_callable=log_retraining_complete,
        provide_context=True,
    )

    # ----------------------------------------------------------------
    # Task dependencies — linear pipeline
    #
    # validate → train → promote → update_baseline → log_complete
    #
    # Each step depends on the previous one succeeding.
    # If validate fails, training never starts.
    # If training fails, promotion never runs.
    # If promotion fails, baseline is not updated.
    # ----------------------------------------------------------------
    validate >> train >> promote >> update_baseline >> log_complete
    