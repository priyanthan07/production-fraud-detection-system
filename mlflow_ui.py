import subprocess
import sys
import yaml
from pathlib import Path

def main():
    config_path = Path("configs/model_config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        uri = config.get("mlflow_tracking_uri", "sqlite:///mlflow.db")
    else:
        uri = "sqlite:///mlflow.db"

    print(f"Starting MLflow UI against: {uri}")
    print("Open: http://localhost:5000")

    subprocess.run([
        sys.executable, "-m", "mlflow", "ui",
        "--backend-store-uri", uri,
        "--host", "0.0.0.0",
        "--port", "5000",
    ])

if __name__ == "__main__":
    main()