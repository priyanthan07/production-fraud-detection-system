-- docker/init-db.sql
-- Creates the Airflow database alongside the default mlflow_tracking database.
-- This script runs automatically on first PostgreSQL container startup.
-- The mlflow_tracking database is already created via POSTGRES_DB env var.

CREATE DATABASE airflow;