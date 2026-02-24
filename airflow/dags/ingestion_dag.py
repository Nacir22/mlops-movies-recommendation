"""
airflow/dags/ingestion_dag.py
DAG 1 — Ingestion des données MovieLens → MinIO + préparation des features.

Planification : quotidien à 02h00 UTC
Pipeline :
  validate_raw_data
       ↓
  upload_to_minio
       ↓
  prepare_features     ← data_loader.prepare_and_save()
       ↓
  check_data_quality
       ↓
  trigger_feast_dag    ← déclenche feast_materialization_dag
"""
from __future__ import annotations

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

log = logging.getLogger(__name__)

# ── Paramètres par défaut ──────────────────────────────────────────────────
DEFAULT_ARGS = {
    "owner":             "mlops-team",
    "depends_on_past":   False,
    "start_date":        datetime(2024, 1, 1),
    "email":             ["mlops-alerts@movielens.com"],
    "email_on_failure":  True,
    "email_on_retry":    False,
    "retries":           2,
    "retry_delay":       timedelta(minutes=5),
}

RAW_DIR  = Path("/opt/airflow/data/raw")
PROC_DIR = Path("/opt/airflow/data/processed")


# ── Fonctions des tâches ───────────────────────────────────────────────────

def validate_raw_data(**context):
    """
    Vérifie que les 3 fichiers .dat existent et ont le bon format.
    Pousse le chemin des données dans XCom pour les tâches suivantes.
    """
    required = ["movies.dat", "users.dat", "ratings.dat"]
    missing  = [f for f in required if not (RAW_DIR / f).exists()]

    if missing:
        raise FileNotFoundError(
            f"Fichiers manquants dans {RAW_DIR} : {missing}\n"
            "Assurez-vous que les fichiers MovieLens sont montés dans /opt/airflow/data/raw/"
        )

    # Vérification du format ratings.dat
    with open(RAW_DIR / "ratings.dat", encoding="latin-1") as f:
        first = f.readline().strip().split("::")
        if len(first) != 4:
            raise ValueError(
                f"Format ratings.dat invalide — attendu 4 champs, trouvé {len(first)}: {first}"
            )

    # Statistiques de base
    n_ratings = sum(1 for _ in open(RAW_DIR / "ratings.dat", encoding="latin-1"))
    n_movies  = sum(1 for _ in open(RAW_DIR / "movies.dat",  encoding="latin-1"))
    n_users   = sum(1 for _ in open(RAW_DIR / "users.dat",   encoding="latin-1"))

    stats = {"n_movies": n_movies, "n_users": n_users, "n_ratings": n_ratings}
    log.info("Données validées : %s", stats)

    context["ti"].xcom_push(key="raw_stats", value=stats)
    context["ti"].xcom_push(key="raw_dir",   value=str(RAW_DIR))
    return stats


def upload_to_minio(**context):
    """
    Upload les fichiers .dat vers MinIO (bucket raw-data).
    Utilise boto3 avec le endpoint MinIO (compatible S3).
    """
    import boto3
    from botocore.exceptions import ClientError

    endpoint   = os.getenv("MINIO_ENDPOINT",    "http://minio:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY",   "minioadmin")
    secret_key = os.getenv("MINIO_SECRET_KEY",   "minioadmin")
    bucket     = os.getenv("MINIO_BUCKET_DATA",  "raw-data")

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
    )

    # Créer le bucket si absent
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        s3.create_bucket(Bucket=bucket)
        log.info("Bucket créé : %s", bucket)

    date_prefix = datetime.now().strftime("%Y/%m/%d")
    uploaded = []

    for filename in ["movies.dat", "users.dat", "ratings.dat"]:
        local_path = RAW_DIR / filename
        s3_key     = f"movielens/{date_prefix}/{filename}"
        s3.upload_file(str(local_path), bucket, s3_key)
        uploaded.append(s3_key)
        log.info("Uploadé : s3://%s/%s", bucket, s3_key)

    context["ti"].xcom_push(key="s3_prefix",  value=f"movielens/{date_prefix}")
    context["ti"].xcom_push(key="s3_uploaded", value=uploaded)
    return {"bucket": bucket, "prefix": date_prefix, "files": uploaded}


def prepare_features(**context):
    """
    Construit la matrice de features via data_loader.prepare_and_save().
    Sauvegarde les CSV dans /opt/airflow/data/processed/.
    """
    import sys
    sys.path.insert(0, "/opt/airflow")
    from training.data_loader import prepare_and_save

    PROC_DIR.mkdir(parents=True, exist_ok=True)

    result = prepare_and_save(
        test_size=float(os.getenv("TEST_SIZE", 0.2)),
        random_state=int(os.getenv("RANDOM_STATE", 42)),
    )

    log.info(
        "Features préparées : %d train | %d test | %d features",
        result["n_train"], result["n_test"], result["n_features"],
    )
    context["ti"].xcom_push(key="feature_stats", value={
        "n_train":    result["n_train"],
        "n_test":     result["n_test"],
        "n_features": result["n_features"],
        "positive_rate": result["positive_rate"],
    })
    return result


def check_data_quality(**context):
    """
    Contrôles qualité sur les données préparées :
    - Pas de colonnes entièrement nulles
    - Taux de positifs dans une plage raisonnable [10%, 90%]
    - Nombre minimum de lignes
    """
    import pandas as pd

    X_train = pd.read_csv(PROC_DIR / "X_train.csv")
    y_train = pd.read_csv(PROC_DIR / "y_train.csv").squeeze()

    checks = {}

    # 1. Taille minimale
    checks["min_rows"] = len(X_train) >= 1000
    if not checks["min_rows"]:
        raise ValueError(f"Dataset trop petit : {len(X_train)} lignes (min 1000)")

    # 2. Pas de colonnes 100% nulles
    null_cols = X_train.columns[X_train.isnull().all()].tolist()
    checks["no_null_columns"] = len(null_cols) == 0
    if not checks["no_null_columns"]:
        raise ValueError(f"Colonnes entièrement nulles : {null_cols}")

    # 3. Taux de positifs équilibré
    pos_rate = float(y_train.mean())
    checks["positive_rate_ok"] = 0.10 <= pos_rate <= 0.90
    if not checks["positive_rate_ok"]:
        raise ValueError(f"Taux de positifs anormal : {pos_rate:.1%}")

    # 4. Pas de doublons
    n_dupes = X_train.duplicated().sum()
    checks["no_duplicates"] = n_dupes < len(X_train) * 0.01   # < 1%

    log.info(
        "Qualité OK — %d lignes | %d features | %.1f%% positifs | %d doublons",
        len(X_train), X_train.shape[1], pos_rate * 100, n_dupes,
    )
    context["ti"].xcom_push(key="quality_checks", value=checks)
    return checks


# ── Définition du DAG ──────────────────────────────────────────────────────

with DAG(
    dag_id="movielens_ingestion",
    default_args=DEFAULT_ARGS,
    description="Ingestion MovieLens → MinIO + préparation features",
    schedule="0 2 * * *",     # 02h00 UTC tous les jours
    catchup=False,
    max_active_runs=1,
    tags=["movielens", "ingestion", "periode-4"],
) as dag:

    t_validate = PythonOperator(
        task_id="validate_raw_data",
        python_callable=validate_raw_data,
    )

    t_upload = PythonOperator(
        task_id="upload_to_minio",
        python_callable=upload_to_minio,
    )

    t_prepare = PythonOperator(
        task_id="prepare_features",
        python_callable=prepare_features,
    )

    t_quality = PythonOperator(
        task_id="check_data_quality",
        python_callable=check_data_quality,
    )

    t_trigger_feast = TriggerDagRunOperator(
        task_id="trigger_feast_materialization",
        trigger_dag_id="feast_materialization",
        wait_for_completion=False,   # Ne pas bloquer l'ingestion
        conf={"triggered_by": "ingestion_dag"},
    )

    # ── Dépendances ────────────────────────────────────────────────
    t_validate >> t_upload >> t_prepare >> t_quality >> t_trigger_feast
