"""
training/data_loader.py
Chargement et préparation des données MovieLens 1M.
Utilisé par les notebooks, train.py et les composants Kubeflow.
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR  = Path(__file__).parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Seuil pour considérer qu'un film est "aimé"
LIKE_THRESHOLD = int(os.getenv("RATING_THRESHOLD", 4))


# ── Chargement des fichiers bruts ──────────────────────────────────────────

def load_movies(path: Path = None) -> pd.DataFrame:
    """
    Charge movies.dat → DataFrame avec one-hot des genres et année extraite.
    Colonnes résultantes : MovieID, Title, Genres, Year, Action, Drama, ...
    """
    path = path or RAW_DIR / "movies.dat"
    df = pd.read_csv(path, sep="::", engine="python", encoding="latin-1",
                     names=["MovieID", "Title", "Genres"])
    # Extraire l'année depuis le titre "(1999)"
    df["Year"] = df["Title"].str.extract(r"\((\d{4})\)").astype(float)
    # One-hot des genres (séparés par "|")
    genres_ohe = df["Genres"].str.get_dummies(sep="|")
    df = pd.concat([df, genres_ohe], axis=1)
    return df


def load_users(path: Path = None) -> pd.DataFrame:
    """
    Charge users.dat → DataFrame avec encodage du genre.
    Colonnes résultantes : UserID, Gender, Age, Occupation, Zip, Gender_enc
    """
    path = path or RAW_DIR / "users.dat"
    df = pd.read_csv(path, sep="::", engine="python", encoding="latin-1",
                     names=["UserID", "Gender", "Age", "Occupation", "Zip"])
    df["Gender_enc"] = (df["Gender"] == "M").astype(int)
    return df


def load_ratings(path: Path = None) -> pd.DataFrame:
    """
    Charge ratings.dat → DataFrame avec variable cible binaire 'Liked'.
    Liked = 1 si Rating >= LIKE_THRESHOLD (défaut : 4).
    """
    path = path or RAW_DIR / "ratings.dat"
    df = pd.read_csv(path, sep="::", engine="python", encoding="latin-1",
                     names=["UserID", "MovieID", "Rating", "Timestamp"])
    df["Liked"] = (df["Rating"] >= LIKE_THRESHOLD).astype(int)
    return df


# ── Construction de la matrice de features ─────────────────────────────────

def build_feature_matrix(
    ratings:  pd.DataFrame,
    users:    pd.DataFrame,
    movies:   pd.DataFrame,
) -> pd.DataFrame:
    """
    Fusionne ratings + users + movies en une matrice de features.

    Features de base :
      - Utilisateur : Gender_enc, Age, Occupation
      - Film        : Year + genres (one-hot)
    Cible : Liked (binaire)
    """
    genre_cols  = [c for c in movies.columns
                   if c not in ["MovieID", "Title", "Genres", "Year"]]
    movie_feats = movies[["MovieID", "Year"] + genre_cols]
    user_feats  = users[["UserID", "Gender_enc", "Age", "Occupation"]]

    df = (ratings[["UserID", "MovieID", "Rating", "Liked"]]
          .merge(user_feats,  on="UserID",  how="left")
          .merge(movie_feats, on="MovieID", how="left"))

    return df.fillna(0)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Retourne les colonnes de features (hors IDs et cibles)."""
    exclude = {"UserID", "MovieID", "Rating", "Liked", "Timestamp"}
    return [c for c in df.columns if c not in exclude]


# ── Pipeline complet : chargement → split → sauvegarde ────────────────────

def prepare_and_save(
    test_size:    float = 0.2,
    random_state: int   = 42,
    extra_stats:  bool  = True,
) -> dict:
    """
    Pipeline complet :
      1. Charge les 3 fichiers .dat
      2. Construit les features (base + stats agrégées si extra_stats=True)
      3. Split stratifié train/test
      4. Sauvegarde en CSV dans data/processed/

    Retourne un dict avec les chemins et les métriques du dataset.
    """
    from sklearn.model_selection import train_test_split

    print("📥 Chargement des données brutes...")
    movies  = load_movies()
    users   = load_users()
    ratings = load_ratings()
    print(f"   {len(movies)} films | {len(users)} utilisateurs | {len(ratings)} notes")

    print("🔧 Construction des features...")
    df = build_feature_matrix(ratings, users, movies)

    if extra_stats:
        # Stats par utilisateur (sévérité, activité)
        user_stats = (ratings.groupby("UserID")
                      .agg(user_avg_rating=("Rating","mean"),
                           user_n_ratings =("Rating","count"),
                           user_std_rating=("Rating","std"))
                      .reset_index().fillna(0))
        # Stats par film (popularité, note moyenne)
        movie_stats = (ratings.groupby("MovieID")
                       .agg(movie_avg_rating=("Rating","mean"),
                            movie_n_ratings =("Rating","count"))
                       .reset_index())
        df = df.merge(user_stats,  on="UserID",  how="left")
        df = df.merge(movie_stats, on="MovieID", how="left")
        df = df.fillna(0)

    # Feature columns
    stat_feats = (["user_avg_rating","user_n_ratings","user_std_rating",
                   "movie_avg_rating","movie_n_ratings"] if extra_stats else [])
    base_feats   = [c for c in get_feature_columns(df) if c not in stat_feats]
    feature_cols = base_feats + stat_feats

    X = df[feature_cols]
    y = df["Liked"]

    print(f"   {len(feature_cols)} features | {y.mean():.1%} positifs (Liked=1)")

    print("✂️  Split train / test (stratifié)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Sauvegarde CSV
    paths = {}
    for name, data in [
        ("X_train", X_train), ("X_test", X_test),
        ("y_train", y_train.to_frame()), ("y_test", y_test.to_frame()),
    ]:
        p = PROC_DIR / f"{name}.csv"
        data.to_csv(p, index=False)
        paths[name] = str(p)

    # Sauvegarder la liste des features
    fc_path = PROC_DIR / "feature_columns.json"
    with open(fc_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    paths["feature_columns"] = str(fc_path)

    print("✅ Données sauvegardées dans data/processed/")
    for k, v in paths.items():
        print(f"   {k:20s} → {v}")

    return {
        "paths": paths,
        "n_train": len(X_train),
        "n_test":  len(X_test),
        "n_features": len(feature_cols),
        "positive_rate": float(y.mean()),
        "feature_columns": feature_cols,
    }


if __name__ == "__main__":
    result = prepare_and_save()
    print(f"\nDataset prêt : {result['n_train']} train | {result['n_test']} test")
