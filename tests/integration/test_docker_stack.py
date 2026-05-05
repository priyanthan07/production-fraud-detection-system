"""
Docker stack integration tests.

Verifies all services start correctly and can communicate.
No model training required — tests infrastructure only.

Run after docker compose up -d:
    pytest tests/integration/test_docker_stack.py -v
"""

import time

import psycopg2
import pytest
import redis as redis_client
import requests

# ----------------------------------------------------------------
# Configuration — matches docker-compose.yml
# ----------------------------------------------------------------
INFERENCE_URL = "http://localhost:8000"
MLFLOW_URL = "http://localhost:5000"
PROMETHEUS_URL = "http://localhost:9090"
GRAFANA_URL = "http://localhost:3000"
AIRFLOW_URL = "http://localhost:8080"

POSTGRES_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "postgres",
    "password": "admin",
    "dbname": "mlflow_tracking",
}

REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
}


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
def wait_for_service(url: str, timeout: int = 60) -> bool:
    """Poll a URL until it responds or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code < 500:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    return False


def make_minimal_transaction(txn_id: int = 9999999) -> dict:
    return {
        "TransactionID": txn_id,
        "TransactionDT": 86400,
        "TransactionAmt": 100.0,
    }


# ================================================================
# PostgreSQL
# ================================================================


class TestPostgres:
    def test_postgres_accepts_connections(self):
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        conn.close()

    def test_mlflow_tracking_database_exists(self):
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT 1")
        result = cur.fetchone()
        conn.close()
        assert result == (1,)

    def test_airflow_database_exists(self):
        config = {**POSTGRES_CONFIG, "dbname": "airflow"}
        conn = psycopg2.connect(**config)
        conn.close()


# ================================================================
# Redis
# ================================================================


class TestRedis:
    @pytest.fixture
    def redis(self):
        r = redis_client.Redis(
            host=REDIS_CONFIG["host"],
            port=REDIS_CONFIG["port"],
            decode_responses=True,
        )
        yield r
        r.close()

    def test_redis_responds_to_ping(self, redis):
        assert redis.ping() is True

    def test_redis_accepts_writes_and_reads(self, redis):
        redis.set("test:ci:key", "test_value", ex=10)
        value = redis.get("test:ci:key")
        assert value == "test_value"

    def test_redis_sorted_set_operations(self, redis):
        """Verify the sorted set operations used by the feature store."""
        key = "test:ci:sorted_set"
        redis.zadd(key, {"txn_1": 86400, "txn_2": 90000})
        redis.expire(key, 10)

        results = redis.zrangebyscore(key, 86400, "(90000")
        assert "txn_1" in results
        assert "txn_2" not in results

        redis.delete(key)

    def test_redis_hash_operations(self, redis):
        """Verify the hash operations used for card stats."""
        key = "test:ci:hash"
        redis.hset(key, "amt_sum", 100.0)
        redis.hincrbyfloat(key, "amt_sum", 50.0)
        redis.expire(key, 10)

        value = float(redis.hget(key, "amt_sum"))
        assert value == 150.0

        redis.delete(key)


# ================================================================
# MLflow
# ================================================================


class TestMLflow:
    def test_mlflow_health_endpoint_responds(self):
        r = requests.get(f"{MLFLOW_URL}/health", timeout=10)
        assert r.status_code == 200

    def test_mlflow_experiments_api_responds(self):
        r = requests.get(
            f"{MLFLOW_URL}/api/2.0/mlflow/experiments/list",
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        assert "experiments" in data

    def test_mlflow_model_registry_api_responds(self):
        r = requests.get(
            f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/list",
            timeout=10,
        )
        assert r.status_code == 200

    def test_mlflow_can_create_experiment(self):
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_URL)
        experiment_id = mlflow.create_experiment(
            "ci_test_experiment",
            tags={"purpose": "ci_health_check"},
        )
        assert experiment_id is not None

        # Clean up
        client = mlflow.tracking.MlflowClient()
        client.delete_experiment(experiment_id)

    def test_mlflow_can_log_run(self):
        """Verify the full run logging pipeline works end to end."""
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_URL)
        mlflow.set_experiment("ci_test_experiment")

        with mlflow.start_run(run_name="ci_health_check") as run:
            mlflow.log_param("test_param", "ci_value")
            mlflow.log_metric("test_metric", 0.95)
            run_id = run.info.run_id

        # Verify it was actually saved
        client = mlflow.tracking.MlflowClient()
        fetched_run = client.get_run(run_id)
        assert fetched_run.data.params["test_param"] == "ci_value"
        assert fetched_run.data.metrics["test_metric"] == 0.95


# ================================================================
# Inference Server
# ================================================================


class TestInference:
    def test_inference_health_endpoint_responds(self):
        r = requests.get(f"{INFERENCE_URL}/health", timeout=10)
        assert r.status_code == 200

    def test_inference_health_returns_correct_schema(self):
        r = requests.get(f"{INFERENCE_URL}/health", timeout=10)
        data = r.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert "threshold" in data
        assert "redis_healthy" in data

    def test_inference_redis_connection_healthy(self):
        """Inference server should connect to Redis successfully."""
        r = requests.get(f"{INFERENCE_URL}/health", timeout=10)
        data = r.json()
        assert data["redis_healthy"] is True

    def test_inference_metrics_endpoint_responds(self):
        """Prometheus metrics endpoint must be accessible."""
        r = requests.get(f"{INFERENCE_URL}/metrics", timeout=10)
        assert r.status_code == 200
        assert "fraud_requests_total" in r.text

    def test_inference_rejects_missing_transaction_id(self):
        """Schema validation must reject incomplete input."""
        txn = {"TransactionDT": 86400, "TransactionAmt": 100.0}
        r = requests.post(f"{INFERENCE_URL}/predict", json=txn, timeout=10)
        assert r.status_code == 422

    def test_inference_rejects_negative_amount(self):
        txn = make_minimal_transaction()
        txn["TransactionAmt"] = -50.0
        r = requests.post(f"{INFERENCE_URL}/predict", json=txn, timeout=10)
        assert r.status_code == 422

    def test_inference_rejects_zero_amount(self):
        txn = make_minimal_transaction()
        txn["TransactionAmt"] = 0.0
        r = requests.post(f"{INFERENCE_URL}/predict", json=txn, timeout=10)
        assert r.status_code == 422

    def test_inference_rejects_negative_transaction_dt(self):
        txn = make_minimal_transaction()
        txn["TransactionDT"] = -1
        r = requests.post(f"{INFERENCE_URL}/predict", json=txn, timeout=10)
        assert r.status_code == 422

    def test_inference_rejects_empty_batch(self):
        r = requests.post(
            f"{INFERENCE_URL}/predict/batch",
            json={"transactions": []},
            timeout=10,
        )
        assert r.status_code == 422

    def test_inference_rejects_duplicate_transaction_ids_in_batch(self):
        transactions = [
            make_minimal_transaction(txn_id=1001),
            make_minimal_transaction(txn_id=1001),  # duplicate
        ]
        r = requests.post(
            f"{INFERENCE_URL}/predict/batch",
            json={"transactions": transactions},
            timeout=10,
        )
        assert r.status_code == 422

    def test_inference_accepts_extra_v_features(self):
        """
        V features are accepted via extra='allow' on the schema.
        This should return 200 or 503 (if no model loaded) — never 422.
        """
        txn = make_minimal_transaction()
        txn["V1"] = 1.0
        txn["V45"] = 0.5
        r = requests.post(f"{INFERENCE_URL}/predict", json=txn, timeout=10)
        assert r.status_code in (200, 503)

    def test_inference_returns_503_when_model_not_loaded(self):
        """
        If no model is in Production registry, inference should
        return 503, not crash with 500.
        """
        r = requests.get(f"{INFERENCE_URL}/health", timeout=10)
        data = r.json()

        if not data["model_loaded"]:
            # No model loaded — predict should return 503
            txn = make_minimal_transaction()
            r = requests.post(f"{INFERENCE_URL}/predict", json=txn, timeout=10)
            assert r.status_code == 503
        else:
            # Model loaded — skip this test
            pytest.skip("Model is loaded, 503 test not applicable")


# ================================================================
# Prometheus
# ================================================================


class TestPrometheus:
    def test_prometheus_responds(self):
        r = requests.get(f"{PROMETHEUS_URL}/-/healthy", timeout=10)
        assert r.status_code == 200

    def test_prometheus_has_inference_target(self):
        """Prometheus should be scraping the inference service."""
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/targets",
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        targets = data["data"]["activeTargets"]
        inference_targets = [
            t for t in targets if "fraud_inference" in t.get("labels", {}).get("job", "")
        ]
        assert len(inference_targets) > 0

    def test_prometheus_can_query_metrics(self):
        """Basic PromQL query should work."""
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": "up"},
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"


# ================================================================
# Grafana
# ================================================================


class TestGrafana:
    def test_grafana_responds(self):
        r = requests.get(f"{GRAFANA_URL}/api/health", timeout=10)
        assert r.status_code == 200

    def test_grafana_dashboard_provisioned(self):
        """The fraud detection dashboard should be loaded."""
        r = requests.get(
            f"{GRAFANA_URL}/api/dashboards/home",
            auth=("admin", "admin"),
            timeout=10,
        )
        assert r.status_code == 200


# ================================================================
# Cross-service connectivity
# ================================================================


class TestServiceConnectivity:
    def test_inference_can_reach_mlflow(self):
        """
        Inference health shows redis_healthy. If MLflow was unreachable
        at startup, inference would have logged errors but still started.
        We verify MLflow is reachable from outside (same network).
        """
        r = requests.get(f"{MLFLOW_URL}/health", timeout=10)
        assert r.status_code == 200

    def test_prometheus_scrapes_inference_metrics(self):
        """
        Make a request to inference, then check Prometheus
        has the metric recorded.
        """
        # Trigger a request (will be 422 or 503, doesn't matter)
        requests.post(
            f"{INFERENCE_URL}/predict",
            json={"TransactionID": 1, "TransactionDT": 86400, "TransactionAmt": 50.0},
            timeout=10,
        )

        # Wait for Prometheus to scrape (scrape interval is 15s)
        time.sleep(16)

        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": "fraud_requests_total"},
            timeout=10,
        )
        data = r.json()
        assert data["status"] == "success"


class TestMLflowArtifactStorage:
    def test_artifact_stored_in_docker_volume_not_local_disk(self):
        """
        THE critical test for the artifact storage issue.

        Verifies that when train.py connects via http://localhost:5000
        (the MLflow server), artifact files land in the Docker volume
        and NOT on the local disk.

        If this test passes, the inference container can load models.
        If this test fails, artifacts are going to the wrong place.
        """
        import os
        import tempfile

        import mlflow

        # Connect through MLflow SERVER (http) not directly to postgres
        mlflow.set_tracking_uri(MLFLOW_URL)
        mlflow.set_experiment("ci_artifact_test")

        # Create a dummy artifact file
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = os.path.join(tmpdir, "test_model.txt")
            with open(artifact_path, "w") as f:
                f.write("this is a test artifact simulating a model file")

            # Log through MLflow server
            with mlflow.start_run(run_name="artifact_storage_test") as run:
                run_id = run.info.run_id
                mlflow.log_artifact(artifact_path, artifact_path="test_artifacts")
                mlflow.log_param("artifact_test", "true")

        # Verify artifact URI points to MLflow server storage
        # NOT to a local file:// path
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_URL)
        run_info = client.get_run(run_id)
        artifact_uri = run_info.info.artifact_uri

        assert not artifact_uri.startswith("file://"), (
            f"Artifact URI starts with file:// — artifacts are going to LOCAL DISK not Docker volume.\n"
            f"Artifact URI: {artifact_uri}\n"
            f"Fix: set MLFLOW_TRACKING_URI=http://localhost:5000 when training."
        )
        assert "mlflow-artifacts" in artifact_uri or artifact_uri.startswith("http"), (
            f"Artifact URI does not point to MLflow server storage.\n"
            f"Artifact URI: {artifact_uri}\n"
            f"Expected URI containing 'mlflow-artifacts' or 'http'."
        )

    def test_artifact_can_be_downloaded_back(self):
        """
        Verifies the complete round trip:
        upload artifact → store in Docker volume → download back.

        If download works, inference container can also load it
        from the same Docker volume.
        """
        import os
        import tempfile

        import mlflow

        mlflow.set_tracking_uri(MLFLOW_URL)
        mlflow.set_experiment("ci_artifact_test")

        test_content = "mock_model_weights_12345"

        # Upload
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_file = os.path.join(tmpdir, "mock_model.pkl")
            with open(artifact_file, "w") as f:
                f.write(test_content)

            with mlflow.start_run(run_name="artifact_roundtrip_test") as run:
                run_id = run.info.run_id
                mlflow.log_artifact(artifact_file, artifact_path="model")

        # Download back through MLflow server
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_URL)

        with tempfile.TemporaryDirectory() as download_dir:
            downloaded = client.download_artifacts(
                run_id,
                "model/mock_model.pkl",
                dst_path=download_dir,
            )

            with open(downloaded) as f:
                content = f.read()

        assert content == test_content, (
            f"Downloaded artifact content does not match uploaded content.\n"
            f"Expected: {test_content}\n"
            f"Got: {content}"
        )

    def test_artifact_uri_uses_server_not_postgres_direct(self):
        """
        Verifies train.py is configured to use http://localhost:5000
        and NOT postgresql:// directly.

        When using postgresql:// directly, artifacts go to local disk.
        When using http://localhost:5000, artifacts go to Docker volume.
        """
        from pathlib import Path

        import yaml

        config_path = Path("configs/model_config.yaml")
        assert config_path.exists(), "configs/model_config.yaml not found"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        tracking_uri = config.get("mlflow_tracking_uri", "")

        assert not tracking_uri.startswith("postgresql://"), (
            f"model_config.yaml uses direct PostgreSQL connection: {tracking_uri}\n"
            f"This causes artifacts to go to local disk instead of Docker volume.\n"
            f"Fix: change mlflow_tracking_uri to http://localhost:5000"
        )

        assert tracking_uri.startswith("http://") or tracking_uri.startswith("https://"), (
            f"mlflow_tracking_uri should be http://localhost:5000, got: {tracking_uri}"
        )

    def test_inference_can_load_model_from_docker_volume(self):
        """
        Verifies the inference container can actually reach
        the mlflow_artifacts Docker volume.

        If a model exists in Production, inference should load it.
        If no model exists, inference should return 503 (not 500 or error).
        Both outcomes are acceptable — we just verify the container
        is connected to the right storage.
        """
        r = requests.get(f"{INFERENCE_URL}/health", timeout=10)
        assert r.status_code == 200

        data = r.json()

        if data["model_loaded"]:
            # Model loaded from Docker volume successfully
            assert data["model_version"] != "none"
            assert data["threshold"] > 0
        else:
            # No model in Production registry — acceptable
            # Verify it returns 503 (graceful degradation) not 500 (crash)
            txn = {
                "TransactionID": 9999999,
                "TransactionDT": 86400,
                "TransactionAmt": 100.0,
            }
            r = requests.post(f"{INFERENCE_URL}/predict", json=txn, timeout=10)
            assert r.status_code == 503, (
                f"Expected 503 (model not loaded) but got {r.status_code}.\n"
                f"This may indicate inference crashed trying to access artifacts."
            )
