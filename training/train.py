"""
training/train.py
Script d'entraînement principal du recommandeur Naïve Bayes.

Usage local :
    python training/train.py

Avec paramètres :
    python training/train.py --gaussian-var-smoothing 1e-2 --min-accuracy 0.60

Compatible MLflow (tracking si MLFLOW_TRACKING_URI est défini).
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Import MLflow avec fallback gracieux
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.data_loader import prepare_and_save
from training.models.naive_bayes_model import MovieNaiveBayesRecommender

PROC_DIR   = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ── Arguments CLI ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Entraînement Naïve Bayes — MovieLens")
    p.add_argument("--gaussian-var-smoothing", type=float, default=1e-2)
    p.add_argument("--bernoulli-alpha",        type=float, default=1.0)
    p.add_argument("--gaussian-weight",        type=float, default=0.6)
    p.add_argument("--bernoulli-weight",       type=float, default=0.4)
    p.add_argument("--test-size",              type=float, default=0.2)
    p.add_argument("--random-state",           type=int,   default=42)
    p.add_argument("--min-accuracy",           type=float, default=0.60,
                   help="Seuil minimal d'accuracy pour valider le modèle")
    p.add_argument("--experiment-name",        type=str,
                   default=os.getenv("MLFLOW_EXPERIMENT_NAME", "movielens-recommender"))
    return p.parse_args()


# ── Métriques ──────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)),                      4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)),    4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)),        4),
        "f1_score":  round(float(f1_score(y_true, y_pred, zero_division=0)),            4),
        "roc_auc":   round(float(roc_auc_score(y_true, y_proba[:, 1])),                4),
    }


# ── Pipeline d'entraînement ────────────────────────────────────────────────

def train(args):
    print("=" * 60)
    print("🎬  Naïve Bayes Recommender — MovieLens")
    print("=" * 60)

    # 1. Préparer les données si nécessaire
    if not (PROC_DIR / "X_train.csv").exists():
        print("\n📥 Préparation des données...")
        prepare_and_save(test_size=args.test_size, random_state=args.random_state)

    X_train = pd.read_csv(PROC_DIR / "X_train.csv")
    X_test  = pd.read_csv(PROC_DIR / "X_test.csv")
    y_train = pd.read_csv(PROC_DIR / "y_train.csv").squeeze()
    y_test  = pd.read_csv(PROC_DIR / "y_test.csv").squeeze()

    with open(PROC_DIR / "feature_columns.json") as f:
        feature_cols = json.load(f)

    print(f"\n📊 Dataset")
    print(f"   Train : {len(X_train):,}  ({y_train.mean():.1%} positifs)")
    print(f"   Test  : {len(X_test):,}   ({y_test.mean():.1%} positifs)")
    print(f"   Features : {len(feature_cols)}")

    # 2. Hyperparamètres
    params = {
        "gaussian_var_smoothing": args.gaussian_var_smoothing,
        "bernoulli_alpha":        args.bernoulli_alpha,
        "gaussian_weight":        args.gaussian_weight,
        "bernoulli_weight":       args.bernoulli_weight,
    }
    print(f"\n⚙️  Hyperparamètres : {params}")

    # 3. Configurer MLflow
    run_id = None
    if MLFLOW_AVAILABLE:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(args.experiment_name)
        run = mlflow.start_run(
            run_name=f"nb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        run_id = run.info.run_id
        print(f"\n🔍 MLflow run : {run_id}")
        print(f"   UI → {tracking_uri}/#/experiments/...")
    else:
        print("\n⚠️  MLflow non disponible — logs locaux uniquement")
        print("   (Lancez docker compose up mlflow pour activer le tracking)")

    # 4. Entraîner
    print("\n🚀 Entraînement...")
    model = MovieNaiveBayesRecommender(**params)
    model.fit(X_train[feature_cols], y_train)
    print("   ✅ Modèle entraîné")

    # 5. Cross-validation
    print("\n📐 Cross-validation (5-fold)...")
    cv_scores = cross_val_score(
        model, X_train[feature_cols], y_train, cv=5, scoring="accuracy"
    )
    cv_mean = round(float(cv_scores.mean()), 4)
    cv_std  = round(float(cv_scores.std()),  4)
    print(f"   CV Accuracy : {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"   Scores      : {[round(s,4) for s in cv_scores]}")

    # 6. Évaluation sur le test set
    print("\n📈 Évaluation test set...")
    y_pred  = model.predict(X_test[feature_cols])
    y_proba = model.predict_proba(X_test[feature_cols])
    metrics = compute_metrics(y_test, y_pred, y_proba)

    print(f"\n{'─'*45}")
    for k, v in metrics.items():
        print(f"   {k:12s} : {v:.4f}")
    print(f"   cv_mean    : {cv_mean:.4f}")
    print(f"   cv_std     : {cv_std:.4f}")
    print(f"{'─'*45}")
    print("\n" + classification_report(y_test, y_pred,
                                       target_names=["Non aimé", "Aimé"]))

    # 7. Logger dans MLflow
    if MLFLOW_AVAILABLE:
        mlflow.log_params(params)
        mlflow.log_param("test_size",    args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("n_features",   len(feature_cols))
        mlflow.log_param("n_train",      len(X_train))
        mlflow.log_param("n_test",       len(X_test))

        mlflow.log_metrics(metrics)
        mlflow.log_metric("cv_accuracy_mean", cv_mean)
        mlflow.log_metric("cv_accuracy_std",  cv_std)

        # Artifacts
        report_path = MODELS_DIR / "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(classification_report(y_test, y_pred,
                                          target_names=["Non aimé", "Aimé"]))
        mlflow.log_artifact(str(report_path))

        fc_path = MODELS_DIR / "feature_columns.json"
        with open(fc_path, "w") as f:
            json.dump(feature_cols, f, indent=2)
        mlflow.log_artifact(str(fc_path))

        # Modèle dans le Registry MLflow
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=os.getenv("MODEL_NAME", "naive-bayes-recommender"),
        )
        mlflow.end_run()
        print(f"✅ Modèle enregistré dans MLflow Registry")

    # 8. Sauvegarde locale (toujours, même avec MLflow)
    bundle = {
        "model":           model,
        "feature_columns": feature_cols,
        "metrics":         metrics,
        "params":          params,
        "run_id":          run_id,
    }
    model_path = MODELS_DIR / "naive_bayes_recommender.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"💾 Modèle sauvegardé : {model_path}")

    # 9. Seuil minimal
    accuracy = metrics["accuracy"]
    if accuracy < args.min_accuracy:
        print(f"\n❌ Accuracy {accuracy:.4f} < seuil {args.min_accuracy} → modèle non promu")
        sys.exit(1)

    print(f"\n✅ Accuracy {accuracy:.4f} ≥ seuil {args.min_accuracy} → modèle validé")
    return {"metrics": metrics, "run_id": run_id, "model_path": str(model_path)}


if __name__ == "__main__":
    args   = parse_args()
    result = train(args)
    print(f"\n🎉 Entraînement terminé — {result['metrics']}")
