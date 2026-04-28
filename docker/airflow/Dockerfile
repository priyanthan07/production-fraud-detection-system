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

# Copy project source code
COPY src/ /opt/airflow/project/src/
COPY configs/ /opt/airflow/project/configs/

# Install the project and all its dependencies from pyproject.toml
RUN pip install --no-cache-dir -e "/opt/airflow/project"

# Copy DAGs into Airflow's DAG directory
COPY airflow/dags/ /opt/airflow/dags/

# Set the project root on PYTHONPATH so DAG tasks can import src.*
ENV PYTHONPATH="/opt/airflow/project:${PYTHONPATH}"

# Expose Airflow webserver port
EXPOSE 8080
