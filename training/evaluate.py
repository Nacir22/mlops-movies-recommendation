"""
training/evaluate.py
Évaluation approfondie du modèle entraîné.
Peut être exécuté seul après train.py.

Usage :
    python training/evaluate.py
"""
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.data_loader import load_movies, load_ratings, load_users

PROC_DIR   = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"


def load_bundle():
    model_path = MODELS_DIR / "naive_bayes_recommender.pkl"
    if not model_path.exists():
        print("❌ Modèle introuvable. Lancez d'abord : python training/train.py")
        sys.exit(1)
    with open(model_path, "rb") as f:
        return pickle.load(f)


def print_metrics(y_test, y_pred, y_proba):
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    print(f"\n✅ Accuracy : {(y_pred == y_test).mean():.4f}")
    print(f"   ROC-AUC : {roc_auc_score(y_test, y_proba[:,1]):.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["Non aimé", "Aimé"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Matrice de confusion :")
    print(f"  TN = {cm[0,0]:5,}   FP = {cm[0,1]:5,}")
    print(f"  FN = {cm[1,0]:5,}   TP = {cm[1,1]:5,}")


def analyse_by_genre(model, X_test, y_test, y_pred, feature_cols):
    """Précision par genre de film."""
    genre_cols = [c for c in feature_cols
                  if c not in ["Gender_enc","Age","Occupation","Year",
                                "user_avg_rating","user_n_ratings","user_std_rating",
                                "movie_avg_rating","movie_n_ratings"]]
    from sklearn.metrics import precision_score
    print("\n📂 Précision par genre (sur le test set) :")
    results = []
    for genre in genre_cols:
        if genre not in X_test.columns:
            continue
        mask = X_test[genre].values == 1
        if mask.sum() < 30:
            continue
        prec = precision_score(y_test[mask], y_pred[mask], zero_division=0)
        results.append((genre, prec, mask.sum()))

    for genre, prec, n in sorted(results, key=lambda x: -x[1])[:12]:
        bar = "█" * int(prec * 25)
        print(f"  {genre:20s} {bar:25s}  {prec:.3f}  (n={n})")


def show_recommendations(model, movies_df, ratings, users):
    """Affiche des exemples de recommandations pour 4 profils."""
    movie_stats = (ratings.groupby("MovieID")
                   .agg(movie_avg_rating=("Rating","mean"),
                        movie_n_ratings =("Rating","count"))
                   .reset_index())
    user_stats  = (ratings.groupby("UserID")
                   .agg(user_avg_rating=("Rating","mean"),
                        user_n_ratings =("Rating","count"),
                        user_std_rating=("Rating","std"))
                   .reset_index().fillna(0))

    genre_cols = [c for c in movies_df.columns
                  if c not in ["MovieID","Title","Genres","Year"]]
    candidates = movies_df[["MovieID","Title","Year"] + genre_cols].copy()

    profiles = [
        {"Gender_enc":1, "Age":25, "Occupation":4,  "label":"👨 Homme 25 ans, ingénieur"},
        {"Gender_enc":0, "Age":35, "Occupation":1,  "label":"👩 Femme 35 ans, développeuse"},
        {"Gender_enc":1, "Age":18, "Occupation":0,  "label":"👦 Homme 18 ans, étudiant"},
        {"Gender_enc":0, "Age":50, "Occupation":7,  "label":"👩 Femme 50 ans, gestionnaire"},
    ]

    print("\n🎬 Exemples de recommandations")
    print("=" * 60)
    for profile in profiles:
        label = profile.pop("label")
        recs = model.recommend_movies(
            profile, candidates,
            user_stats=user_stats, movie_stats=movie_stats,
            top_k=5
        )
        print(f"\n{label} :")
        for _, r in recs.iterrows():
            print(f"   {r.Title:45s}  score={r.score:.3f}")


def main():
    print("📊 Évaluation — MovieLens Naïve Bayes Recommender")
    print("=" * 60)

    bundle       = load_bundle()
    model        = bundle["model"]
    feature_cols = bundle["feature_columns"]

    X_test = pd.read_csv(PROC_DIR / "X_test.csv")
    y_test = pd.read_csv(PROC_DIR / "y_test.csv").squeeze()

    y_pred  = model.predict(X_test[feature_cols])
    y_proba = model.predict_proba(X_test[feature_cols])

    print_metrics(y_test, y_pred, y_proba)
    analyse_by_genre(model, X_test, y_test.values, y_pred, feature_cols)

    movies  = load_movies()
    ratings = load_ratings()
    users   = load_users()
    show_recommendations(model, movies, ratings, users)


if __name__ == "__main__":
    main()
