"""
training/train_with_feast.py
Entraînement du modèle en utilisant le Feature Store Feast
pour récupérer les features (au lieu de les construire manuellement).

Démontre la séparation propre entre :
  - La définition des features (feast/features/)
  - L'entraînement (training/)
  - Le serving (serving/) — utilise le même store en online

Usage :
    python training/train_with_feast.py
    python training/train_with_feast.py --min-accuracy 0.60
"""
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.models.naive_bayes_model import MovieNaiveBayesRecommender
from feast.store.feature_store_local import LocalFeatureStore

PROC_DIR   = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Features demandées au Feature Store
USER_FEATURES = [
    "user_features:gender_enc",
    "user_features:age",
    "user_features:occupation",
    "user_features:user_avg_rating",
    "user_features:user_n_ratings",
    "user_features:user_std_rating",
]
MOVIE_FEATURES = [
    "movie_features:year",
    "movie_features:movie_avg_rating",
    "movie_features:movie_n_ratings",
    "movie_features:action",
    "movie_features:adventure",
    "movie_features:animation",
    "movie_features:childrens",
    "movie_features:comedy",
    "movie_features:crime",
    "movie_features:documentary",
    "movie_features:drama",
    "movie_features:fantasy",
    "movie_features:film_noir",
    "movie_features:horror",
    "movie_features:musical",
    "movie_features:mystery",
    "movie_features:romance",
    "movie_features:sci_fi",
    "movie_features:thriller",
    "movie_features:war",
    "movie_features:western",
]
ALL_FEATURES = USER_FEATURES + MOVIE_FEATURES


def build_training_dataset(store: LocalFeatureStore) -> pd.DataFrame:
    """
    Construit le dataset d'entraînement via le Feature Store.

    Étapes :
    1. Charger les ratings (source de vérité pour les étiquettes)
    2. Construire l'entity_df (UserID, MovieID, event_timestamp)
    3. Appeler get_historical_features() pour enrichir avec les features
    4. Retourner le dataset complet avec les étiquettes
    """
    print("📥 Chargement des ratings (source des étiquettes)...")
    ratings = pd.read_csv("data/raw/ratings.dat", sep="::", engine="python",
                          encoding="latin-1", names=["UserID","MovieID","Rating","Timestamp"])
    ratings["Liked"] = (ratings["Rating"] >= 4).astype(int)

    print(f"   {len(ratings):,} interactions | {ratings.Liked.mean():.1%} positifs")

    # Entity DataFrame : une ligne par interaction
    entity_df = ratings[["UserID","MovieID","Timestamp"]].copy()
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["Timestamp"], unit="s")
    entity_df = entity_df.rename(columns={"UserID":"user_id", "MovieID":"movie_id"})

    print("🔍 Récupération des features depuis le Feature Store...")
    df = store.get_historical_features(
        entity_df=entity_df,
        features=ALL_FEATURES,
    )

    # Réattacher les étiquettes
    df["Liked"] = ratings["Liked"].values
    df = df.fillna(0)

    # Nettoyer les colonnes non-features
    drop_cols = {"user_id", "movie_id", "event_timestamp", "Timestamp",
                 "UserID", "MovieID"}
    feat_cols = [c for c in df.columns
                 if c not in drop_cols and c != "Liked"]

    print(f"✅ Dataset construit : {df.shape} | {len(feat_cols)} features")
    return df, feat_cols


def train_with_feast(args):
    print("=" * 60)
    print("🌾 Entraînement via Feature Store — MovieLens")
    print("=" * 60)

    # 1. Initialiser et matérialiser le Feature Store
    print("\n🔄 Initialisation du Feature Store...")
    store = LocalFeatureStore()
    store.materialize()

    # 2. Construire le dataset d'entraînement
    print("\n📊 Construction du dataset via get_historical_features()...")
    df, feat_cols = build_training_dataset(store)

    # 3. Split
    from sklearn.model_selection import train_test_split
    X = df[feat_cols]
    y = df["Liked"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n   Train : {len(X_train):,} | Test : {len(X_test):,}")

    # 4. Entraîner
    print("\n🚀 Entraînement du modèle...")
    model = MovieNaiveBayesRecommender(
        gaussian_var_smoothing=1e-2,
        bernoulli_alpha=1.0,
        gaussian_weight=0.6,
        bernoulli_weight=0.4,
    )
    model.fit(X_train, y_train)

    # 5. Évaluer
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    from sklearn.model_selection import cross_val_score

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    cv      = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

    metrics = {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)),               4),
        "f1_score":  round(float(f1_score(y_test, y_pred, zero_division=0)),     4),
        "roc_auc":   round(float(roc_auc_score(y_test, y_proba[:, 1])),          4),
        "cv_mean":   round(float(cv.mean()),                                     4),
        "cv_std":    round(float(cv.std()),                                      4),
    }

    print(f"\n📈 Résultats :")
    print(f"{'─'*40}")
    for k, v in metrics.items():
        print(f"   {k:12s} : {v:.4f}")
    print(f"{'─'*40}")

    # 6. Sauvegarder
    bundle = {
        "model":           model,
        "feature_columns": feat_cols,
        "metrics":         metrics,
        "source":          "feast",
        "feast_features":  ALL_FEATURES,
    }
    model_path = MODELS_DIR / "naive_bayes_feast.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n💾 Modèle sauvegardé : {model_path}")

    # 7. Valider le seuil
    if metrics["accuracy"] < args.min_accuracy:
        print(f"\n❌ Accuracy {metrics['accuracy']} < seuil {args.min_accuracy}")
        sys.exit(1)

    print(f"✅ Modèle validé (accuracy ≥ {args.min_accuracy})")
    return bundle


def demo_online_inference(store: LocalFeatureStore = None):
    """
    Démontre l'inférence temps réel via l'online store.
    Simule ce que fera l'API FastAPI à chaque requête.
    """
    if store is None:
        store = LocalFeatureStore()

    print("\n" + "=" * 60)
    print("⚡ DÉMO : Inférence temps réel via Online Store")
    print("=" * 60)

    # Charger le modèle entraîné avec Feast
    model_path = MODELS_DIR / "naive_bayes_feast.pkl"
    if not model_path.exists():
        print("❌ Modèle non trouvé. Lancez d'abord train_with_feast.py")
        return

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    model     = bundle["model"]
    feat_cols = bundle["feature_columns"]

    # Simuler une requête : user_id=1 veut savoir s'il aimera les films 1, 5, 10
    user_id   = 1
    movie_ids = [1, 5, 10, 15, 20]

    print(f"\n👤 Requête : user_id={user_id}, films candidats={movie_ids}")
    print("   → get_online_features() depuis Redis (ou cache local)...")

    entity_rows = [{"user_id": user_id, "movie_id": mid} for mid in movie_ids]

    response = store.get_online_features(
        features=ALL_FEATURES,
        entity_rows=entity_rows,
    )

    feat_df = response.to_df()

    # Aligner les colonnes avec le modèle
    for c in feat_cols:
        if c not in feat_df.columns:
            feat_df[c] = 0.0
    feat_df = feat_df[feat_cols].fillna(0)

    probas = model.predict_proba(feat_df)[:, 1]

    print(f"\n{'─'*45}")
    print(f"{'MovieID':>10}  {'P(Liked)':>10}  {'Recommandé':>12}")
    print(f"{'─'*45}")
    for mid, p in zip(movie_ids, probas):
        rec = "✅ OUI" if p >= 0.5 else "❌ NON"
        print(f"{mid:>10}  {p:>10.3f}  {rec:>12}")
    print(f"{'─'*45}")
    print(f"\nLatence simulée : < 5ms (lecture Redis/cache)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-accuracy", type=float, default=0.60)
    args = parser.parse_args()

    bundle = train_with_feast(args)

    store = LocalFeatureStore()
    store.materialize()
    demo_online_inference(store)

    print("\n🎉 Pipeline Feast complet !")
    print("   offline → materialize → online → inférence temps réel")
