.PHONY: help install test lint serve train promote mlflow-ui \
        docker-up docker-down docker-build clean drift retrain \
        bootstrap-redis feast-apply feast-materialize feast-ui

help:
	@echo "============================================"
	@echo "  Fraud Detection System — Make Targets"
	@echo "============================================"
	@echo ""
	@echo "  Development:"
	@echo "    install              Install project dependencies with uv"
	@echo "    test                 Run all tests"
	@echo "    test-unit            Run unit tests only"
	@echo "    test-int             Run integration tests only"
	@echo "    lint                 Run ruff linter"
	@echo ""
	@echo "  ML Pipeline:"
	@echo "    train                Run the full training pipeline"
	@echo "    promote              Promote latest model to Production"
	@echo "    drift                Run drift detection"
	@echo "    retrain              Force retrain regardless of drift"
	@echo ""
	@echo "  Feature Store:"
	@echo "    feast-apply          Register feature definitions with Feast"
	@echo "    feast-materialize    Push batch features from parquet to Redis via Feast"
	@echo "    bootstrap-redis      Warm Redis velocity/aggregation features from training data"
	@echo "    feast-ui             Open Feast UI in browser"
	@echo ""
	@echo "  Serving:"
	@echo "    serve                Start FastAPI inference server"
	@echo "    mlflow-ui            Start MLflow tracking UI"
	@echo ""
	@echo "  Docker:"
	@echo "    docker-build         Build all Docker images"
	@echo "    docker-up            Start full stack (docker compose up)"
	@echo "    docker-down          Stop full stack (docker compose down)"
	@echo "    docker-logs          Tail logs from all services"
	@echo ""
	@echo "  Cleanup:"
	@echo "    clean                Remove cached files and artifacts"
	@echo ""

install:
	uv sync

test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v --tb=short

test-int:
	pytest tests/integration/ -v --tb=short

lint:
	ruff check src/ tests/
	ruff format src/ tests/ --check

train:
	python -m src.training.train

promote:
	python -m src.registry.model_manager

drift:
	python -m src.monitoring.drift_detector

retrain:
	python -m src.retraining.trigger --force

feast-apply:
	python scripts/feast_apply.py

feast-materialize:
	python scripts/feast_materialize.py

bootstrap-redis:
	python scripts/bootstrap_redis.py

feast-ui:
	cd feature_repo && feast ui

serve:
	uvicorn src.inference.app:app --host 0.0.0.0 --port 8000

mlflow-ui:
	python mlflow_ui.py

docker-build:
	docker compose build

docker-up:
	docker compose up -d
	@echo ""
	@echo "Services starting..."
	@echo "  MLflow UI:    http://localhost:5000"
	@echo "  Inference:    http://localhost:8000/docs"
	@echo "  Prometheus:   http://localhost:9090"
	@echo "  Grafana:      http://localhost:3000"
	@echo "  Airflow:      http://localhost:8080"
	@echo "  Redis:        localhost:6379"
	@echo ""

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache .mypy_cache
	