"""
pipelines/pipeline.py
Pipeline Kubeflow Pipelines — MovieLens Naïve Bayes Recommender.

Assemble les 3 composants en un pipeline bout-en-bout :
  prepare_data → train_model → evaluate_model

Usage :
    # Compiler le pipeline en YAML (pour Kubeflow)
    python pipelines/pipeline.py --compile

    # Exécuter localement sans Kubeflow
    python pipelines/pipeline.py --local

    # Soumettre à un cluster Kubeflow
    python pipelines/pipeline.py --submit --host http://kubeflow.local:8888

Architecture :
  ┌─────────────────────────────────────────────────────────┐
  │  prepare_data_component                                  │
  │    → X_train, X_test, y_train, y_test, feature_cols     │
  │           ↓                                             │
  │  train_model_component                                   │
  │    → model, cv_metrics                                  │
  │           ↓                                             │
  │  evaluate_model_component                                │
  │    → test_metrics, confusion_matrix                     │
  │    → ✅ si accuracy ≥ min_accuracy, sinon ❌ raise      │
  └─────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Pipeline KFP (si kfp installé) ────────────────────────────────────────

def build_kfp_pipeline():
    """Construit et retourne le pipeline KFP compilable."""
    from kfp import dsl
    from pipelines.components.prepare_data  import prepare_data_component
    from pipelines.components.train_model   import train_model_component
    from pipelines.components.evaluate_model import evaluate_model_component

    @dsl.pipeline(
        name="movielens-naive-bayes-pipeline",
        description="Pipeline d'entraînement Naïve Bayes pour MovieLens 1M",
    )
    def movielens_pipeline(
        # Hyperparamètres du modèle
        gaussian_var_smoothing: float = 1e-2,
        bernoulli_alpha:        float = 1.0,
        gaussian_weight:        float = 0.6,
        bernoulli_weight:       float = 0.4,
        # Split
        test_size:              float = 0.2,
        random_state:           int   = 42,
        # Validation
        min_accuracy:           float = 0.60,
        # Infrastructure
        mlflow_tracking_uri: str = "http://mlflow:5000",
        experiment_name:     str = "movielens-recommender",
        minio_endpoint:      str = "http://minio:9000",
        minio_bucket:        str = "raw-data",
    ):
        # Étape 1 : Préparer les données
        prep_task = prepare_data_component(
            test_size=test_size,
            random_state=random_state,
            minio_endpoint=minio_endpoint,
            minio_bucket=minio_bucket,
        )
        prep_task.set_display_name("📂 Préparer les données")
        prep_task.set_cpu_request("0.5")
        prep_task.set_memory_request("512Mi")

        # Étape 2 : Entraîner le modèle
        train_task = train_model_component(
            gaussian_var_smoothing=gaussian_var_smoothing,
            bernoulli_alpha=bernoulli_alpha,
            gaussian_weight=gaussian_weight,
            bernoulli_weight=bernoulli_weight,
            mlflow_tracking_uri=mlflow_tracking_uri,
            experiment_name=experiment_name,
            x_train_in=prep_task.outputs["x_train_out"],
            y_train_in=prep_task.outputs["y_train_out"],
            feature_cols_in=prep_task.outputs["feature_cols_out"],
        )
        train_task.set_display_name("🚀 Entraîner le modèle")
        train_task.set_cpu_request("1.0")
        train_task.set_memory_request("1Gi")

        # Étape 3 : Évaluer et valider
        eval_task = evaluate_model_component(
            min_accuracy=min_accuracy,
            mlflow_tracking_uri=mlflow_tracking_uri,
            x_test_in=prep_task.outputs["x_test_out"],
            y_test_in=prep_task.outputs["y_test_out"],
            model_input=train_task.outputs["model_output"],
            feature_cols_in=prep_task.outputs["feature_cols_out"],
        )
        eval_task.set_display_name("📊 Évaluer le modèle")
        eval_task.set_cpu_request("0.5")
        eval_task.set_memory_request("512Mi")
        eval_task.after(train_task)

    return movielens_pipeline


def compile_pipeline(output_path: str = "pipelines/pipeline.yaml"):
    """Compile le pipeline en YAML pour Kubeflow."""
    from kfp import compiler
    pipeline_fn = build_kfp_pipeline()
    compiler.Compiler().compile(pipeline_func=pipeline_fn, package_path=output_path)
    print(f"✅ Pipeline compilé : {output_path}")
    return output_path


def submit_pipeline(
    host:         str   = "http://localhost:8888",
    run_name:     str   = None,
    pipeline_file: str  = "pipelines/pipeline.yaml",
    arguments:    dict  = None,
):
    """Soumet le pipeline à un cluster Kubeflow."""
    import kfp
    from datetime import datetime

    run_name  = run_name or f"movielens_nb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    arguments = arguments or {}

    client = kfp.Client(host=host)
    run    = client.create_run_from_pipeline_package(
        pipeline_file=pipeline_file,
        arguments=arguments,
        run_name=run_name,
    )
    print(f"✅ Pipeline soumis — run_id : {run.run_id}")
    print(f"   UI : {host}/#/runs/details/{run.run_id}")
    return run.run_id


# ── Simulateur local (sans KFP) ───────────────────────────────────────────

class LocalPipelineRunner:
    """
    Exécute le pipeline sans Kubeflow.
    Reproduit la séquence des composants avec leurs entrées/sorties.
    Utilisé pour le développement local et les tests CI.
    """

    def __init__(
        self,
        gaussian_var_smoothing: float = 1e-2,
        bernoulli_alpha:        float = 1.0,
        gaussian_weight:        float = 0.6,
        bernoulli_weight:       float = 0.4,
        test_size:              float = 0.2,
        random_state:           int   = 42,
        min_accuracy:           float = 0.60,
        data_dir:               str   = "data/processed",
        output_dir:             str   = "models",
    ):
        self.params = dict(
            gaussian_var_smoothing=gaussian_var_smoothing,
            bernoulli_alpha=bernoulli_alpha,
            gaussian_weight=gaussian_weight,
            bernoulli_weight=bernoulli_weight,
            test_size=test_size,
            random_state=random_state,
            min_accuracy=min_accuracy,
        )
        self.data_dir   = data_dir
        self.output_dir = output_dir

    def run(self) -> dict:
        from pipelines.components.prepare_data   import prepare_data_local
        from pipelines.components.train_model    import train_model_local
        from pipelines.components.evaluate_model import evaluate_model_local

        print("=" * 60)
        print("🚀 LocalPipelineRunner — MovieLens Naïve Bayes")
        print("=" * 60)
        print(f"Paramètres : {self.params}")
        print()

        # ── Étape 1 ────────────────────────────────────────────────
        print("─" * 40)
        print("ÉTAPE 1 : Préparation des données")
        print("─" * 40)
        prep_out = prepare_data_local(
            test_size=self.params["test_size"],
            random_state=self.params["random_state"],
            output_dir=self.data_dir,
        )

        # ── Étape 2 ────────────────────────────────────────────────
        print()
        print("─" * 40)
        print("ÉTAPE 2 : Entraînement du modèle")
        print("─" * 40)
        train_out = train_model_local(
            gaussian_var_smoothing=self.params["gaussian_var_smoothing"],
            bernoulli_alpha=self.params["bernoulli_alpha"],
            gaussian_weight=self.params["gaussian_weight"],
            bernoulli_weight=self.params["bernoulli_weight"],
            data_dir=self.data_dir,
            output_dir=self.output_dir,
        )

        # ── Étape 3 ────────────────────────────────────────────────
        print()
        print("─" * 40)
        print("ÉTAPE 3 : Évaluation du modèle")
        print("─" * 40)
        eval_out = evaluate_model_local(
            min_accuracy=self.params["min_accuracy"],
            data_dir=self.data_dir,
            model_path=train_out["model_path"],
        )

        result = {
            "status":       "Succeeded",
            "train_metrics": train_out["metrics"],
            "eval_metrics":  eval_out,
            "model_path":    train_out["model_path"],
        }

        print()
        print("=" * 60)
        print("✅ Pipeline terminé avec succès")
        print(f"   CV accuracy  : {train_out['metrics']['cv_accuracy_mean']:.4f} "
              f"± {train_out['metrics']['cv_accuracy_std']:.4f}")
        print(f"   Test accuracy: {eval_out['test_accuracy']:.4f}")
        print(f"   Test ROC-AUC : {eval_out['test_roc_auc']:.4f}")
        print("=" * 60)

        return result


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline Kubeflow — MovieLens")
    parser.add_argument("--compile",  action="store_true", help="Compiler en YAML")
    parser.add_argument("--local",    action="store_true", help="Exécuter localement")
    parser.add_argument("--submit",   action="store_true", help="Soumettre à Kubeflow")
    parser.add_argument("--host",     default="http://localhost:8888")
    parser.add_argument("--gaussian-var-smoothing", type=float, default=1e-2)
    parser.add_argument("--bernoulli-alpha",         type=float, default=1.0)
    parser.add_argument("--gaussian-weight",         type=float, default=0.6)
    parser.add_argument("--bernoulli-weight",        type=float, default=0.4)
    parser.add_argument("--min-accuracy",            type=float, default=0.60)
    args = parser.parse_args()

    if args.compile:
        try:
            compile_pipeline()
        except ImportError:
            print("❌ kfp non installé. pip install kfp")
            sys.exit(1)

    elif args.local:
        runner = LocalPipelineRunner(
            gaussian_var_smoothing=args.gaussian_var_smoothing,
            bernoulli_alpha=args.bernoulli_alpha,
            gaussian_weight=args.gaussian_weight,
            bernoulli_weight=args.bernoulli_weight,
            min_accuracy=args.min_accuracy,
        )
        runner.run()

    elif args.submit:
        try:
            if not Path("pipelines/pipeline.yaml").exists():
                compile_pipeline()
            submit_pipeline(host=args.host)
        except ImportError:
            print("❌ kfp non installé. pip install kfp")
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
