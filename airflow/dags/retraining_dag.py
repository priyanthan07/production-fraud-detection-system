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
        f"Data validation passed. {len(raw_df)} rows, {len(raw_df.columns)} columns."
    )

    # Push row count to XCom for logging
    context["ti"].xcom_push(key="row_count", value=len(raw_df))

def prepare_training_data(**context):
    """
    Task 2: Merge original training features with new labeled production data.

    Flow:
    1. Load scored_features.parquet (features computed at inference time)
    2. Load labels.parquet (confirmed fraud/not-fraud from analysts/chargebacks)
    3. Inner join on TransactionID → rows that have both features AND labels
    4. Merge with original train_features.parquet
    5. Apply sliding window: drop rows older than 6 months
    6. Save as new train_features.parquet for training

    If no labeled production data exists, training uses original data only.
    This task never fails — it falls back gracefully.
    """
    import logging
    import sys
    from pathlib import Path

    import pandas as pd

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    sys.path.insert(0, "/opt/airflow/project")

    project_root          = Path("/opt/airflow/project")
    scored_features_path  = project_root / "data/production/scored_features.parquet"
    labels_path           = project_root / "data/production/labels.parquet"
    original_train_path   = project_root / "data/processed/train_features.parquet"

    # Sliding window: 6 months in seconds
    SLIDING_WINDOW_SECONDS = 180 * 86400

    if not original_train_path.exists():
        logger.error(
            "Original train_features.parquet not found. "
            "Run training pipeline first."
        )
        raise FileNotFoundError(str(original_train_path))

    original = pd.read_parquet(original_train_path)
    logger.info(f"Original training data: {len(original):,} rows")

    # Check if labeled production data is available
    if not scored_features_path.exists():
        logger.info(
            "No scored_features.parquet found. "
            "Training on original data only."
        )
        context["ti"].xcom_push(key="new_labeled_count", value=0)
        return

    if not labels_path.exists():
        logger.info(
            "No labels.parquet found. "
            "Labels have not been collected yet. "
            "Training on original data only."
        )
        context["ti"].xcom_push(key="new_labeled_count", value=0)
        return

    # Load and join scored features with confirmed labels
    scored = pd.read_parquet(scored_features_path)
    labels = pd.read_parquet(labels_path)

    logger.info(f"Scored production rows: {len(scored):,}")
    logger.info(f"Confirmed labels: {len(labels):,}")

    # Inner join: only rows that have both features AND a confirmed label
    # Labels file must have columns: TransactionID, isFraud
    if "TransactionID" not in labels.columns or "isFraud" not in labels.columns:
        logger.error(
            "labels.parquet must contain TransactionID and isFraud columns. "
            "Skipping new labeled data."
        )
        context["ti"].xcom_push(key="new_labeled_count", value=0)
        return

    new_labeled = scored.merge(
        labels[["TransactionID", "isFraud"]],
        on="TransactionID",
        how="inner",
    )

    if len(new_labeled) == 0:
        logger.info(
            "No labeled production rows match scored features. "
            "Training on original data only."
        )
        context["ti"].xcom_push(key="new_labeled_count", value=0)
        return

    logger.info(f"New labeled rows for training: {len(new_labeled):,}")

    # Ensure new labeled data has the same columns as original
    # Add missing columns as NaN, drop extra columns
    original_cols = set(original.columns)
    new_cols      = set(new_labeled.columns)

    # Columns in original but not in new → fill with NaN
    for col in original_cols - new_cols:
        new_labeled[col] = float("nan")

    # Align column order to match original
    new_labeled = new_labeled[list(original.columns)]

    # Combine original + new labeled
    combined = pd.concat([original, new_labeled], ignore_index=True)
    combined = combined.sort_values("TransactionDT").reset_index(drop=True)

    logger.info(f"Combined before sliding window: {len(combined):,} rows")

    # Apply sliding window: keep only last 6 months
    max_dt  = combined["TransactionDT"].max()
    cutoff  = max_dt - SLIDING_WINDOW_SECONDS
    combined = combined[combined["TransactionDT"] >= cutoff].reset_index(drop=True)

    logger.info(
        f"After 6-month sliding window: {len(combined):,} rows "
        f"(dropped {len(original) + len(new_labeled) - len(combined):,} old rows)"
    )

    # Save combined as the new training dataset
    combined.to_parquet(original_train_path, index=False)
    logger.info(f"Saved combined training data to {original_train_path}")

    context["ti"].xcom_push(key="new_labeled_count", value=len(new_labeled))

def run_training(**context):
    import logging
    import subprocess
    import sys
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    new_labeled_count = context["ti"].xcom_pull(
        task_ids="prepare_training_data",
        key="new_labeled_count",
    ) or 0
    logger.info(f"Training with {new_labeled_count:,} new labeled rows merged in.")

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
            f"Training pipeline failed with return code {result.returncode}"
        )

    logger.info("Training pipeline completed successfully.")
    context["ti"].xcom_push(key="training_output", value=result.stdout[-3000:])


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
        logger.info(f"Model version {result['version']} promoted to Production!")
    else:
        logger.warning(
            f"Model version {result['version']} failed quality gates. "
            f"Current Production model remains unchanged."
        )

    context["ti"].xcom_push(
        key="promotion_result",
        value={
            "promoted": result["promoted"],
            "version": result["version"],
            "gates_passed": result["gates_passed"],
        },
    )


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
    
    new_labeled_count = context["ti"].xcom_pull(
        task_ids="prepare_training_data",
        key="new_labeled_count",
    ) or 0

    logger.info("=" * 50)
    logger.info("Retraining Pipeline Complete")
    logger.info("=" * 50)
    logger.info(f"  New labeled rows used: {new_labeled_count:,}")

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
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["fraud", "ml", "retraining"],
) as dag:

    validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
        provide_context=True,
    )

    prepare_data = PythonOperator(
        task_id="prepare_training_data",
        python_callable=prepare_training_data,
        provide_context=True,
    )

    train = PythonOperator(
        task_id="run_training",
        python_callable=run_training,
        provide_context=True,
        execution_timeout=timedelta(hours=3),
    )

    promote = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model,
        provide_context=True,
    )

    update_baseline = PythonOperator(
        task_id="save_new_baseline",
        python_callable=save_new_baseline,
        provide_context=True,
    )

    log_complete = PythonOperator(
        task_id="log_retraining_complete",
        python_callable=log_retraining_complete,
        provide_context=True,
    )

    # validate → prepare_data → train → promote → update_baseline → log_complete
    validate >> prepare_data >> train >> promote >> update_baseline >> log_complete
