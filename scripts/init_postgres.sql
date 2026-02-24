-- init_postgres.sql
-- Créé automatiquement au premier démarrage de PostgreSQL
-- Crée la base de données mlflow si elle n'existe pas déjà

-- La base 'mlflow' est créée via POSTGRES_DB dans docker-compose.
-- Ce script crée aussi la base 'airflow' et ajoute les extensions utiles.

CREATE USER airflow WITH PASSWORD 'airflow_password';
CREATE DATABASE airflow OWNER airflow;
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Accorder tous les droits à l'utilisateur mlflow
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;
