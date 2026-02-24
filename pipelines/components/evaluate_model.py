"""
pipelines/components/evaluate_model.py
Composant Kubeflow Pipelines — Évaluation et validation du modèle.

Calcule les métriques sur le test set et bloque la promotion
si l'accuracy est inférieure au seuil min_accuracy.
"""
from __future__ import annotations

try:
    from kfp import dsl
    from kfp.dsl import Dataset, Input, Model, Output, Metrics, component, ClassificationMetrics

    @component(
        base_image="python:3.10-slim",
        packages_to_install=[
            "scikit-learn==1.3.2",
            "pandas==2.1.4",
            "numpy==1.26.3",
            "mlflow==2.9.2",
        ],
    )
    def evaluate_model_component(
        min_accuracy:        float,
        mlflow_tracking_uri: str,
        # Entrées
        x_test_in:     Input[Dataset],
        y_test_in:     Input[Dataset],
        model_input:   Input[Model],
        feature_cols_in: Input[Dataset],
        # Sorties
        metrics_output:       Output[Metrics],
        class_metrics_output: Output[ClassificationMetrics],
    ):
        """Composant KFP : évalue et valide le modèle entraîné."""
        import os, json, pickle
        import pandas as pd
        import numpy as np
        from sklearn.metrics import (
            accuracy_score, f1_score, roc_auc_score,
            precision_score, recall_score, confusion_matrix,
            classification_report,
        )

        # Charger
        X_test = pd.read_csv(x_test_in.path)
        y_test = pd.read_csv(y_test_in.path).squeeze()
        with open(feature_cols_in.path) as f:
            feat_cols = json.load(f)

        model_file = os.path.join(model_input.path, "model.pkl")
        with open(model_file, "rb") as f:
            bundle = pickle.load(f)
        model = bundle["model"]

        X_test = X_test[feat_cols]
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        accuracy  = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall    = float(recall_score(y_test, y_pred, zero_division=0))
        f1        = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc   = float(roc_auc_score(y_test, y_proba[:,1]))

        # Logger dans MLflow
        try:
            import mlflow
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            with mlflow.start_run():
                mlflow.log_metrics({
                    "test_accuracy":  accuracy,
                    "test_precision": precision,
                    "test_recall":    recall,
                    "test_f1":        f1,
                    "test_roc_auc":   roc_auc,
                })
        except Exception as e:
            print(f"⚠️  MLflow : {e}")

        # Métriques KFP
        metrics_output.log_metric("test_accuracy",  accuracy)
        metrics_output.log_metric("test_precision", precision)
        metrics_output.log_metric("test_recall",    recall)
        metrics_output.log_metric("test_f1",        f1)
        metrics_output.log_metric("test_roc_auc",   roc_auc)

        # Matrice de confusion KFP
        cm     = confusion_matrix(y_test, y_pred)
        labels = ["Non aimé", "Aimé"]
        class_metrics_output.log_confusion_matrix(labels, cm.tolist())

        print(classification_report(y_test, y_pred, target_names=labels))
        print(f"Accuracy : {accuracy:.4f} | ROC-AUC : {roc_auc:.4f}")

        # Validation du seuil — bloque le pipeline si non atteint
        if accuracy < min_accuracy:
            raise ValueError(
                f"❌ Accuracy {accuracy:.4f} < seuil {min_accuracy} — "
                "modèle non promu. Ajustez les hyperparamètres."
            )

        print(f"✅ Validation réussie — accuracy {accuracy:.4f} ≥ {min_accuracy}")

    KFP_AVAILABLE = True

except ImportError:
    KFP_AVAILABLE = False
    evaluate_model_component = None


# ── Implémentation locale ──────────────────────────────────────────────────

def evaluate_model_local(
    min_accuracy: float = 0.60,
    data_dir:     str   = "data/processed",
    model_path:   str   = "models/kfp_model.pkl",
) -> dict:
    """Équivalent local de evaluate_model_component."""
    import sys, json, pickle
    from pathlib import Path
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
        precision_score, recall_score, classification_report,
    )

    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").squeeze()
    with open(f"{data_dir}/feature_columns.json") as f:
        feat_cols = json.load(f)

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]

    y_pred  = model.predict(X_test[feat_cols])
    y_proba = model.predict_proba(X_test[feat_cols])

    metrics = {
        "test_accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "test_precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "test_recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "test_f1":        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "test_roc_auc":   round(float(roc_auc_score(y_test, y_proba[:,1])), 4),
    }

    print(classification_report(y_test, y_pred, target_names=["Non aimé","Aimé"]))
    for k, v in metrics.items():
        print(f"  {k:18s} : {v:.4f}")

    if metrics["test_accuracy"] < min_accuracy:
        raise ValueError(
            f"❌ test_accuracy {metrics['test_accuracy']} < {min_accuracy}"
        )

    print(f"\n✅ Évaluation OK — accuracy {metrics['test_accuracy']:.4f} ≥ {min_accuracy}")
    return metrics
