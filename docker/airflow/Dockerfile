FROM apache/airflow:2.9.3-python3.12

# Switch to root to install system packages
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Install project Python dependencies into Airflow's environment
COPY pyproject.toml /opt/airflow/project/pyproject.toml

RUN pip install --no-cache-dir \
    "mlflow>=2.10.0,<3.0.0" \
    "psycopg2-binary>=2.9.0" \
    "protobuf>=3.20.0,<4.0.0" \
    "sqlalchemy==1.4.46" \
    "xgboost>=2.0.0" \
    "lightgbm>=4.0.0" \
    "catboost>=1.2.0" \
    "shap>=0.43.0" \
    "pandas>=2.0.0" \
    "numpy>=1.24.0" \
    "scikit-learn>=1.3.0" \
    "pyyaml>=6.0" \
    "optuna>=3.0.0" \
    "pyarrow>=14.0.0"\
    "redis>=5.0.0" \
    "feast[redis]>=0.36.0" \

# Copy project source code
COPY src/ /opt/airflow/project/src/
COPY configs/ /opt/airflow/project/configs/

# Copy DAGs into Airflow's DAG directory
COPY airflow/dags/ /opt/airflow/dags/

# Set the project root on PYTHONPATH so DAG tasks can import src.*
ENV PYTHONPATH="/opt/airflow/project:${PYTHONPATH}"

# Expose Airflow webserver port
EXPOSE 8080
