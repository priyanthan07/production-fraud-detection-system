import logging
import mlflow
import yaml
import sys
from mlflow.tracking import MlflowClient
from pathlib import Path

logger = logging.getLogger(__name__)

QUALITY_GATES = {
    "auc_roc" : 0.85,
    "auc_pr" : 0.40,
    "recall" : 0.30,
    "precision" : 0.30,
}

MODEL_NAME = "fraud_detection_model"

def get_client() -> MlflowClient:
    """
        Return a configured MLflow client.
        Reads the tracking URI from model_config.yaml so it stays
        consistent with train.py and never needs to be hardcoded here.
    """
    config_path = Path("configs/model_config.yaml")
    
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        tracking_uri = config.get(
            "mlflow_tracking_uri",
            "postgresql://postgres:admin@localhost:5432/mlflow_tracking",
        )
    else:
        tracking_uri = (
            "postgresql://postgres:admin@localhost:5432/mlflow_tracking"
        )
        
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()

def get_latest_version(client: MlflowClient, stage: str = "None") -> object:
    """
        Return the most recently registered model version in the given stage.
    """
    versions = client.get_latest_versions(MODEL_NAME, stages=[stage])
    if not versions:
        return None
    
    return max(versions, key=lambda v: int(v.version))

def get_all_versions(client: MlflowClient) -> list:
    """
        Return all registered versions of the model, sorted by version number descending (newest first).
    """
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    return sorted(versions, key=lambda v: int(v.version), reverse=True)

def get_run_metrics(client: MlflowClient, run_id: str) -> dict:
    """
        Fetch the metrics logged during the training run that produced this model version.
    """
    run = client.get_run(run_id)
    return run.data.metrics

def get_run_params(client: MlflowClient, run_id: str) -> dict:
    """ 
        Fetch the params logged during the training run.
    """
    run = client.get_run(run_id)
    return run.data.params

def check_quality_gates(metrics: dict) -> tuple:
    """
        Run all quality gates against the model's logged metrics.
    """
    report = {}
    all_passed = True
    
    for metric_name, min_value in QUALITY_GATES.items():
        actual_value = metrics.get(metric_name)
        
        if actual_value is None:
            report[metric_name] = {
                "required" : min_value,
                "actual" : None,
                "passed" : False,
                "reason" : "metric not found in run"
            }
            all_passed = False
            continue
        
        passed = actual_value >= min_value
        report[metric_name] = {
            "required" : min_value,
            "actual" : round(actual_value, 4),
            "passed" : passed,
        }
        
        if not passed:
            all_passed = False
            
    return all_passed, report

def promote_to_staging(client: MlflowClient, version: str) -> None:
    """ 
        Promote a model version from None to Staging.
    """
    logger.info(f"Promoting {MODEL_NAME} version {version} to Staging...")
    
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Staging",
        archive_existing_versions=False,  # we want to keep the previous Staging version visible while we validate the new candidate. We will archive it explicitly if the new one passes gates.
    )
    
    logger.info(f"{MODEL_NAME} version {version} is now in Staging.")
    
def promote_to_production(client: MlflowClient, version: str) -> None:
    """
        Promote a model version from Staging to Production.
    """
    logger.info(f"Promoting {MODEL_NAME} version {version} to Production...")
    
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    
    logger.info(f"{MODEL_NAME} version {version} is now in Production. Previous Production version has been archived.")
    
def reject_staging(
    client: MlflowClient,
    version: str,
    reason: str,
) -> None:
    """ 
        Archive a Staging model that failed quality gates. Adds a rejection tag so the reason is visible in the MLflow UI.
    """
    
    logger.warning(f"{MODEL_NAME} version {version} rejected from Staging. Reason: {reason}")
    
    client.set_model_version_tag(
        name=MODEL_NAME,
        version=version,
        key="rejection_reason",
        value=reason,
    )
    
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Archived",
        archive_existing_versions=False,
    )
    
    logger.info(f"{MODEL_NAME} version {version} archived due to failed gates.")
    
def run_promotion_workflow(
    version: str = None,
    auto_select: bool = True,
) -> dict:
    """ 
        Run the full promotion workflow end to end.
 
        Steps:
        1. Identify which model version to promote (latest if auto_select)
        2. Promote it to Staging
        3. Fetch its training metrics from the MLflow run
        4. Run quality gates
        5a. If gates pass  → promote to Production
        5b. If gates fail  → archive with rejection reason
    """
    client = get_client()
    
    # Step 1: Identify the version to promote
    if version is None and auto_select:
        latest = get_latest_version(client, stage="None")
        if latest is None:
            raise ValueError(
                "No model versions in 'None' stage found. Run training first to register a new model version."
            )
            
        version = latest.version
        logger.info(f"Auto-selected latest version: {MODEL_NAME} v{version}")
        
    elif version is None:
        raise ValueError("Either provide a version number or set auto_select=True.")
    
    # Fetch the version object to get the run_id
    version_obj = client.get_model_version(MODEL_NAME, version)
    run_id = version_obj.run_id
    
    logger.info(f"Evaluating {MODEL_NAME} version {version} (run_id: {run_id[:8]}...)")
    
    # Step 2: Promote to Staging
    promote_to_staging(client, version)
    
    # Step 3: Fetch metrics from the training run
    metrics = get_run_metrics(client, run_id)
    params = get_run_params(client, run_id)
    
    logger.info("Metrics from training run:")
    for name, value in metrics.items():
        if name in QUALITY_GATES:
            logger.info(f"  {name}: {value:.4f} (gate requires >= {QUALITY_GATES[name]})")
            
    # Step 4: Run quality gates
    gates_passed, gate_report = check_quality_gates(metrics)
    
    logger.info("Quality gate results:")
    
    for gate_name, result in gate_report.items():
        status = "PASS" if result["passed"] else "FAIL"
        logger.info(f"  [{status}] {gate_name}: {result['actual']} >= {result['required']} required")
        
    # Step 5: Promote or reject
    if gates_passed:
        promote_to_production(client, version)
        promoted = True
        logger.info(f"Promotion successful. {MODEL_NAME} version {version} is now serving traffic.")
        
    else:
        failed_gates = [name for name, r in gate_report.items() if not r["passed"]]
        
        reason = (
            f"Failed quality gates: {', '.join(failed_gates)}. "
            + " | ".join(
                f"{name}: got {gate_report[name]['actual']}, "
                f"needed >= {gate_report[name]['required']}"
                for name in failed_gates
            )
        )
        reject_staging(client, version, reason)
        promoted = False
        logger.warning(
            f"Promotion failed. {MODEL_NAME} version {version} did not meet quality gates and has been archived."
        )
    
    return {
        "version": version,
        "run_id": run_id,
        "promoted": promoted,
        "gates_passed": gates_passed,
        "gate_report": gate_report,
        "metrics": metrics,
        "optimal_threshold": float(
            params.get("optimal_threshold", 0.5)
        ),
    }
    
def get_production_model_info() -> dict:
    """ 
        Return information about the currently active Production model.
 
        Used by the inference server at startup to know which model
        to load and what threshold to use.
    """
    client = get_client()
    production = get_latest_version(client, stage="Production")
    
    if production is None:
        logger.warning("No model currently in Production stage.")
        return None
    
    run_id = production.run_id
    metrics = get_run_metrics(client, run_id)
    params = get_run_params(client, run_id)
    
    info = {
        "version": production.version,
        "run_id": run_id,
        "model_uri": f"models:/{MODEL_NAME}/Production",
        "optimal_threshold": float(params.get("optimal_threshold", 0.5)),
        "metrics": metrics,
    }
 
    logger.info(f"Production model: {MODEL_NAME} version {production.version}")
    logger.info(f"  AUC-ROC: {metrics.get('auc_roc', 'N/A')}")
    logger.info(f"  AUC-PR:  {metrics.get('auc_pr', 'N/A')}")
    logger.info(f"  Threshold: {info['optimal_threshold']}")
 
    return info

def list_all_versions() -> None:
    """
        Print a summary table of all registered model versions.
        Useful for debugging and auditing.
    """
    client = get_client()
    versions = get_all_versions(client)
 
    if not versions:
        logger.info(f"No versions registered for {MODEL_NAME}.")
        return
 
    logger.info(f"\nAll versions of {MODEL_NAME}:")
    logger.info(f"{'Version':<10} {'Stage':<15} {'Run ID':<12} {'Created'}")
    logger.info("-" * 60)
 
    for v in versions:
        run_id_short = v.run_id[:8] + "..."
        logger.info(
            f"{v.version:<10} {v.current_stage:<15} "
            f"{run_id_short:<12} {v.creation_timestamp}"
        )
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        list_all_versions()
        
    elif len(sys.argv) > 1 and sys.argv[1] == "info":
        info = get_production_model_info()
        if info:
            print(info)
            
    else:
        # Default: run the full promotion workflow
        result = run_promotion_workflow(auto_select=True)
        print("\nPromotion result:")
        print(f"  Version:  {result['version']}")
        print(f"  Promoted: {result['promoted']}")
        print(f"  Threshold: {result['optimal_threshold']}")
    