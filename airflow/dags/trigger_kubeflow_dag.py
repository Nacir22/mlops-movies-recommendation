"""
airflow/dags/trigger_kubeflow_dag.py
DAG 3 — Déclenchement hebdomadaire du pipeline d'entraînement Kubeflow.

Planification : lundi à 04h00 UTC (après ingestion + matérialisation)

Pipeline :
  check_drift_threshold
          ↓ (si drift > seuil)        ↓ (sinon)
  trigger_training_pipeline      skip_training
          ↓                               ↓
  wait_for_pipeline_completion         (fin)
          ↓
  notify_completion
"""
from __future__ import annotations

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

log = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner":            "mlops-team",
    "depends_on_past":  False,
    "start_date":       datetime(2024, 1, 1),
    "email_on_failure": True,
    "retries":          1,
    "retry_delay":      timedelta(minutes=10),
}

DRIFT_THRESHOLD  = float(os.getenv("DRIFT_THRESHOLD", 0.3))
MIN_ACCURACY     = float(os.getenv("MIN_ACCURACY",    0.60))
KUBEFLOW_HOST    = os.getenv("KUBEFLOW_HOST",         "http://localhost:8888")
REPORTS_DIR      = Path("/opt/airflow/monitoring/evidently/reports")


def check_drift_threshold(**context):
    """
    Lit le dernier rapport Evidently (JSON) et décide si ré-entraînement nécessaire.
    Si aucun rapport → déclenche par précaution.

    Branche vers :
      "trigger_training_pipeline"  si drift > seuil OU premier run
      "skip_training"              sinon
    """
    import json, glob

    report_files = sorted(REPORTS_DIR.glob("drift_report_*.json")) if REPORTS_DIR.exists() else []

    if not report_files:
        log.info("Aucun rapport Evidently trouvé — déclenchement par précaution")
        context["ti"].xcom_push(key="drift_score",  value=-1.0)
        context["ti"].xcom_push(key="drift_reason", value="no_report")
        return "trigger_training_pipeline"

    with open(report_files[-1]) as f:
        report = json.load(f)

    drift_score = report.get("global_score", 0.0)
    context["ti"].xcom_push(key="drift_score",  value=drift_score)
    context["ti"].xcom_push(key="drift_reason", value="evidently_report")

    log.info(
        "Drift score : %.4f (seuil : %.4f) — rapport : %s",
        drift_score, DRIFT_THRESHOLD, report_files[-1].name,
    )

    if drift_score > DRIFT_THRESHOLD:
        log.info("⚠️  Drift détecté → déclenchement du ré-entraînement")
        return "trigger_training_pipeline"
    else:
        log.info("✅ Pas de drift significatif → skip")
        return "skip_training"


def trigger_training_pipeline(**context):
    """
    Soumet le pipeline Kubeflow via le SDK kfp.
    Fallback : loggue la soumission sans échouer si Kubeflow absent.
    """
    drift_score = context["ti"].xcom_pull(key="drift_score")
    log.info("Déclenchement pipeline — drift_score=%.4f", drift_score or -1)

    run_name = f"nb_training_{datetime.now().strftime('%Y%m%d_%H%M')}"

    try:
        import kfp
        client = kfp.Client(host=KUBEFLOW_HOST)
        run = client.create_run_from_pipeline_package(
            pipeline_file="/opt/airflow/pipelines/pipeline.yaml",
            arguments={
                "gaussian_var_smoothing": "1e-2",
                "bernoulli_alpha":        "1.0",
                "gaussian_weight":        "0.6",
                "bernoulli_weight":       "0.4",
                "min_accuracy":           str(MIN_ACCURACY),
            },
            run_name=run_name,
        )
        run_id = run.run_id
        log.info("✅ Pipeline Kubeflow soumis — run_id : %s", run_id)
        context["ti"].xcom_push(key="kubeflow_run_id", value=run_id)

    except ImportError:
        log.warning("kfp non installé — simulation du trigger")
        log.info("  run_name : %s | host : %s", run_name, KUBEFLOW_HOST)
        context["ti"].xcom_push(key="kubeflow_run_id", value=f"simulated_{run_name}")

    except Exception as e:
        log.error("Erreur Kubeflow : %s", e)
        raise

    return run_name


def wait_for_pipeline(**context):
    """
    Attend la fin du pipeline Kubeflow (polling).
    En mode simulé, attend 5s et retourne succès.
    """
    import time

    run_id = context["ti"].xcom_pull(
        task_ids="trigger_training_pipeline", key="kubeflow_run_id"
    )
    log.info("Attente pipeline — run_id : %s", run_id)

    if run_id and run_id.startswith("simulated_"):
        log.info("Mode simulé — pas d'attente réelle")
        time.sleep(2)
        context["ti"].xcom_push(key="pipeline_status", value="Succeeded")
        return "Succeeded"

    try:
        import kfp
        client   = kfp.Client(host=KUBEFLOW_HOST)
        timeout  = 3600          # 1h max
        interval = 30            # Poll toutes les 30s
        elapsed  = 0

        while elapsed < timeout:
            run_detail = client.get_run(run_id)
            status     = run_detail.run.status
            log.info("  Pipeline status : %s (%ds écoulés)", status, elapsed)

            if status in ("Succeeded", "Failed", "Error", "Skipped"):
                context["ti"].xcom_push(key="pipeline_status", value=status)
                if status != "Succeeded":
                    raise RuntimeError(f"Pipeline Kubeflow terminé avec statut : {status}")
                return status

            time.sleep(interval)
            elapsed += interval

        raise TimeoutError(f"Pipeline non terminé après {timeout}s")

    except ImportError:
        log.info("kfp absent — simulation terminée avec succès")
        return "Succeeded"


def notify_completion(**context):
    """Log final avec résumé de l'exécution."""
    drift_score = context["ti"].xcom_pull(key="drift_score")
    run_id      = context["ti"].xcom_pull(
        task_ids="trigger_training_pipeline", key="kubeflow_run_id"
    )
    status = context["ti"].xcom_pull(
        task_ids="wait_for_pipeline", key="pipeline_status"
    )

    summary = {
        "dag_run":    context["run_id"],
        "drift_score": drift_score,
        "kubeflow_run_id": run_id,
        "pipeline_status": status,
        "timestamp": datetime.utcnow().isoformat(),
    }
    log.info("✅ Pipeline terminé : %s", summary)

    # En prod : envoyer vers Slack/PagerDuty/email
    return summary


# ── DAG ───────────────────────────────────────────────────────────────────

with DAG(
    dag_id="trigger_kubeflow_training",
    default_args=DEFAULT_ARGS,
    description="Déclenchement hebdomadaire du pipeline d'entraînement Kubeflow",
    schedule="0 4 * * 1",    # Lundi 04h00 UTC
    catchup=False,
    max_active_runs=1,
    tags=["movielens", "kubeflow", "training", "periode-4"],
) as dag:

    t_check_drift = BranchPythonOperator(
        task_id="check_drift_threshold",
        python_callable=check_drift_threshold,
    )

    t_trigger = PythonOperator(
        task_id="trigger_training_pipeline",
        python_callable=trigger_training_pipeline,
    )

    t_skip = EmptyOperator(
        task_id="skip_training",
    )

    t_wait = PythonOperator(
        task_id="wait_for_pipeline",
        python_callable=wait_for_pipeline,
    )

    t_notify = PythonOperator(
        task_id="notify_completion",
        python_callable=notify_completion,
        trigger_rule="none_failed_min_one_success",
    )

    t_end = EmptyOperator(
        task_id="end",
        trigger_rule="none_failed_min_one_success",
    )

    # ── Dépendances ────────────────────────────────────────────────
    t_check_drift >> [t_trigger, t_skip]
    t_trigger     >> t_wait >> t_notify >> t_end
    t_skip        >> t_end
