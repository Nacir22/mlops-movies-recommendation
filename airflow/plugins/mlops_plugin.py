"""
airflow/plugins/mlops_plugin.py
Plugin Airflow — hooks et opérateurs personnalisés pour le projet MovieLens.

Contient :
  - MinIOHook      : connexion boto3 → MinIO
  - MLflowHook     : connexion mlflow tracking
  - DataQualityOperator : opérateur réutilisable de contrôle qualité
"""
from __future__ import annotations

import os
import logging
from typing import Any

log = logging.getLogger(__name__)


# ── MinIO Hook ─────────────────────────────────────────────────────────────

class MinIOHook:
    """
    Hook boto3 vers MinIO (S3-compatible).
    Peut être enregistré comme Airflow Hook via la classe AirflowPlugin.
    """
    def __init__(
        self,
        endpoint:   str = None,
        access_key: str = None,
        secret_key: str = None,
    ):
        import boto3
        self._client = boto3.client(
            "s3",
            endpoint_url         = endpoint   or os.getenv("MINIO_ENDPOINT",    "http://minio:9000"),
            aws_access_key_id    = access_key or os.getenv("MINIO_ACCESS_KEY",  "minioadmin"),
            aws_secret_access_key= secret_key or os.getenv("MINIO_SECRET_KEY",  "minioadmin"),
            region_name          = "us-east-1",
        )

    def upload_file(self, local_path: str, bucket: str, s3_key: str) -> str:
        self._client.upload_file(local_path, bucket, s3_key)
        log.info("Uploadé : s3://%s/%s", bucket, s3_key)
        return f"s3://{bucket}/{s3_key}"

    def download_file(self, bucket: str, s3_key: str, local_path: str) -> str:
        self._client.download_file(bucket, s3_key, local_path)
        log.info("Téléchargé : s3://%s/%s → %s", bucket, s3_key, local_path)
        return local_path

    def list_objects(self, bucket: str, prefix: str = "") -> list[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
        return keys

    def bucket_exists(self, bucket: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self._client.head_bucket(Bucket=bucket)
            return True
        except ClientError:
            return False

    def ensure_bucket(self, bucket: str) -> bool:
        """Crée le bucket s'il n'existe pas."""
        if not self.bucket_exists(bucket):
            self._client.create_bucket(Bucket=bucket)
            log.info("Bucket créé : %s", bucket)
            return True
        return False


# ── MLflow Hook ────────────────────────────────────────────────────────────

class MLflowHook:
    """Hook MLflow pour requêter le registry depuis les DAGs."""

    def __init__(self, tracking_uri: str = None):
        import mlflow
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://mlflow:5000"
        )
        mlflow.set_tracking_uri(self.tracking_uri)
        self._mlflow = mlflow

    def get_latest_model_metrics(self, model_name: str, stage: str = "Production") -> dict:
        """Récupère les métriques du dernier modèle en production."""
        client = self._mlflow.tracking.MlflowClient()
        try:
            versions = client.get_latest_versions(model_name, stages=[stage])
            if not versions:
                return {}
            version = versions[0]
            run     = client.get_run(version.run_id)
            return {
                "version":  version.version,
                "run_id":   version.run_id,
                "metrics":  run.data.metrics,
                "params":   run.data.params,
                "stage":    stage,
            }
        except Exception as e:
            log.warning("Erreur MLflow : %s", e)
            return {}

    def promote_model(
        self,
        model_name: str,
        version:    int,
        to_stage:   str = "Production",
    ) -> bool:
        """Promeut une version de modèle vers le stage cible."""
        client = self._mlflow.tracking.MlflowClient()
        try:
            client.transition_model_version_stage(
                name=model_name, version=str(version), stage=to_stage
            )
            log.info("✅ Modèle %s v%s → %s", model_name, version, to_stage)
            return True
        except Exception as e:
            log.error("Erreur promotion : %s", e)
            return False


# ── Data Quality Operator ──────────────────────────────────────────────────

class DataQualityCheck:
    """
    Utilitaire de contrôle qualité réutilisable dans les DAGs.
    Utilisé par check_data_quality() dans ingestion_dag.py.
    """

    @staticmethod
    def check_no_nulls(df, threshold: float = 0.05) -> dict:
        """Vérifie que le taux de nulls est < threshold par colonne."""
        null_rates = (df.isnull().sum() / len(df))
        bad_cols   = null_rates[null_rates > threshold].to_dict()
        return {"passed": len(bad_cols) == 0, "bad_columns": bad_cols}

    @staticmethod
    def check_class_balance(y, min_rate: float = 0.10, max_rate: float = 0.90) -> dict:
        """Vérifie que la classe positive est dans [min_rate, max_rate]."""
        pos_rate = float(y.mean())
        return {
            "passed":    min_rate <= pos_rate <= max_rate,
            "pos_rate":  pos_rate,
            "min_rate":  min_rate,
            "max_rate":  max_rate,
        }

    @staticmethod
    def check_min_rows(df, min_rows: int = 1000) -> dict:
        """Vérifie le nombre minimal de lignes."""
        return {"passed": len(df) >= min_rows, "n_rows": len(df), "min_rows": min_rows}

    @classmethod
    def run_all(cls, X, y) -> dict:
        """Lance tous les contrôles et retourne un résumé."""
        checks = {
            "no_nulls":      cls.check_no_nulls(X),
            "class_balance": cls.check_class_balance(y),
            "min_rows":      cls.check_min_rows(X),
        }
        all_passed = all(v["passed"] for v in checks.values())
        failed     = [k for k, v in checks.items() if not v["passed"]]
        return {"all_passed": all_passed, "failed": failed, "details": checks}


# ── Enregistrement Airflow (optionnel) ─────────────────────────────────────
try:
    from airflow.plugins_manager import AirflowPlugin

    class MLOpsPlugin(AirflowPlugin):
        name    = "mlops_plugin"
        hooks   = [MinIOHook, MLflowHook]

except ImportError:
    pass   # Airflow non installé — classes utilisables directement
