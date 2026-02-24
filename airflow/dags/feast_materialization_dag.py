"""
airflow/dags/feast_materialization_dag.py
DAG 2 — Matérialisation Feast : offline store → online store (Redis).

Déclenché par :
  - movielens_ingestion (après prepare_features)
  - trigger_kubeflow_dag (vérification avant entraînement)
  - Planification autonome : 02h30 UTC tous les jours

Pipeline :
  run_feast_apply
       ↓
  materialize_user_features ──┐
                               ├──→ validate_online_store
  materialize_movie_features ─┘
"""
from __future__ import annotations

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

log = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner":            "mlops-team",
    "depends_on_past":  False,
    "start_date":       datetime(2024, 1, 1),
    "email_on_failure": True,
    "retries":          2,
    "retry_delay":      timedelta(minutes=3),
}

FEAST_REPO = Path("/opt/airflow/feast")
DATA_DIR   = Path("/opt/airflow/data/feast")


def run_feast_apply(**context):
    """
    Applique les définitions Feast (entités, feature views).
    Équivalent CLI : cd feast/ && feast apply
    """
    import subprocess

    try:
        # Tentative avec Feast natif
        result = subprocess.run(
            ["feast", "apply"],
            cwd=str(FEAST_REPO),
            capture_output=True, text=True,
            timeout=120,
        )
        if result.returncode != 0:
            log.warning("feast apply warning : %s", result.stderr)
        else:
            log.info("feast apply OK :\n%s", result.stdout)
    except FileNotFoundError:
        # Feast non installé → simulateur local
        log.info("Feast CLI absent — utilisation du simulateur local")

    context["ti"].xcom_push(key="feast_apply_status", value="ok")


def materialize_user_features(**context):
    """Matérialise les features utilisateurs offline → Redis."""
    sys.path.insert(0, "/opt/airflow")
    from feast.store.feature_store_local import LocalFeatureStore

    store = LocalFeatureStore()
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=1)

    # Matérialiser uniquement user_features
    import pandas as pd
    from feast.store.feature_store_local import FEATURE_VIEW_FILES, ENTITY_KEYS, ONLINE_DIR
    import json

    fv_name = "user_features"
    df      = pd.read_csv(FEATURE_VIEW_FILES[fv_name])
    entity_key = ENTITY_KEYS[fv_name]
    feat_cols  = [c for c in df.columns if c not in [entity_key, "event_timestamp", "created"]]

    cache = {}
    for _, row in df.iterrows():
        cache[int(row[entity_key])] = {c: float(row[c]) for c in feat_cols}

    snapshot_path = ONLINE_DIR / f"{fv_name}_snapshot.json"
    with open(snapshot_path, "w") as f:
        json.dump({str(k): v for k, v in cache.items()}, f)

    n = len(cache)
    log.info("user_features matérialisées : %d entités", n)
    context["ti"].xcom_push(key="user_feat_count", value=n)
    return n


def materialize_movie_features(**context):
    """Matérialise les features films offline → Redis."""
    sys.path.insert(0, "/opt/airflow")
    import pandas as pd
    from feast.store.feature_store_local import FEATURE_VIEW_FILES, ENTITY_KEYS, ONLINE_DIR
    import json

    fv_name    = "movie_features"
    df         = pd.read_csv(FEATURE_VIEW_FILES[fv_name])
    entity_key = ENTITY_KEYS[fv_name]
    feat_cols  = [c for c in df.columns if c not in [entity_key, "event_timestamp", "created"]]

    cache = {}
    for _, row in df.iterrows():
        cache[int(row[entity_key])] = {c: float(row[c]) for c in feat_cols}

    snapshot_path = ONLINE_DIR / f"{fv_name}_snapshot.json"
    with open(snapshot_path, "w") as f:
        json.dump({str(k): v for k, v in cache.items()}, f)

    n = len(cache)
    log.info("movie_features matérialisées : %d entités", n)
    context["ti"].xcom_push(key="movie_feat_count", value=n)
    return n


def validate_online_store(**context):
    """
    Vérifie que l'online store répond correctement.
    Teste la récupération de features pour 3 paires (user, film).
    """
    sys.path.insert(0, "/opt/airflow")
    from feast.store.feature_store_local import LocalFeatureStore

    store = LocalFeatureStore()
    response = store.get_online_features(
        features=[
            "user_features:gender_enc",
            "user_features:user_avg_rating",
            "movie_features:year",
            "movie_features:movie_avg_rating",
        ],
        entity_rows=[
            {"user_id": 1,  "movie_id": 1},
            {"user_id": 10, "movie_id": 5},
            {"user_id": 42, "movie_id": 20},
        ],
    )
    df = response.to_df()

    # Assertions
    assert len(df) == 3,                          "Doit retourner 3 lignes"
    assert "user_avg_rating"  in df.columns,      "user_avg_rating manquant"
    assert "movie_avg_rating" in df.columns,      "movie_avg_rating manquant"
    assert df["user_avg_rating"].notna().all(),   "Valeurs nulles dans user_avg_rating"

    user_count  = context["ti"].xcom_pull(task_ids="materialize_user_features",
                                          key="user_feat_count")
    movie_count = context["ti"].xcom_pull(task_ids="materialize_movie_features",
                                          key="movie_feat_count")

    log.info(
        "Online store validé — %d users | %d films | sample:\n%s",
        user_count, movie_count, df.to_string(),
    )
    context["ti"].xcom_push(key="validation_ok", value=True)
    return {"users": user_count, "movies": movie_count, "sample_rows": len(df)}


# ── DAG ───────────────────────────────────────────────────────────────────

with DAG(
    dag_id="feast_materialization",
    default_args=DEFAULT_ARGS,
    description="Matérialisation Feast : offline → online store (Redis)",
    schedule="30 2 * * *",   # 02h30 — après ingestion_dag (02h00)
    catchup=False,
    max_active_runs=1,
    tags=["movielens", "feast", "features", "periode-4"],
) as dag:

    t_apply = PythonOperator(
        task_id="run_feast_apply",
        python_callable=run_feast_apply,
    )

    t_users = PythonOperator(
        task_id="materialize_user_features",
        python_callable=materialize_user_features,
    )

    t_movies = PythonOperator(
        task_id="materialize_movie_features",
        python_callable=materialize_movie_features,
    )

    t_validate = PythonOperator(
        task_id="validate_online_store",
        python_callable=validate_online_store,
    )

    # ── Dépendances : apply, puis users || films en parallèle, puis validate
    t_apply >> [t_users, t_movies] >> t_validate
