import subprocess
import yaml
from pathlib import Path

with open("configs/model_config.yaml") as f:
    config = yaml.safe_load(f)

tracking_uri = config["mlflow_tracking_uri"]

print(f"Starting MLflow UI with backend: {tracking_uri}")
print("Open http://localhost:5000 in your browser")
print("Press Ctrl+C to stop")

mlflow_exe = Path(".venv/Scripts/mlflow.exe")

subprocess.run(
    [
        str(mlflow_exe),
        "ui",
        "--backend-store-uri",
        tracking_uri,
        "--port",
        "5000",
    ]
)
