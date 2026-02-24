"""
pipelines/components/prepare_data.py
Composant Kubeflow Pipelines — Préparation des données MovieLens.

Ce composant :
  1. Charge les fichiers .dat depuis MinIO (ou local en dev)
  2. Construit la matrice de features via data_loader
  3. Effectue le split train/test stratifié
  4. Sauvegarde X_train, X_test, y_train, y_test en tant qu'artefacts KFP

Compatible avec kfp v2 (google.cloud.aiplatform ou kubeflow natif).
En local, fonctionne via le simulateur LocalPipeline.
"""
from __future__ import annotations

# ── Définition KFP (active si kfp installé) ────────────────────────────────
try:
    from kfp import dsl
    from kfp.dsl import Dataset, Output, component

    @component(
        base_image="python:3.10-slim",
        packages_to_install=[
            "scikit-learn==1.3.2",
            "pandas==2.1.4",
            "numpy==1.26.3",
            "boto3==1.28.85",
        ],
    )
    def prepare_data_component(
        test_size:    float,
        random_state: int,
        minio_endpoint: str,
        minio_bucket:   str,
        # Sorties — artefacts Dataset KFP
        x_train_out: Output[Dataset],
        x_test_out:  Output[Dataset],
        y_train_out: Output[Dataset],
        y_test_out:  Output[Dataset],
        feature_cols_out: Output[Dataset],
    ):
        """Composant KFP : prépare et sauvegarde les datasets d'entraînement."""
        import sys, os, json
        import pandas as pd
        import numpy as np
        from pathlib import Path
        from sklearn.model_selection import train_test_split

        # Télécharger les données depuis MinIO si disponible
        raw_dir = Path("/tmp/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)

        try:
            import boto3
            s3 = boto3.client(
                "s3",
                endpoint_url=minio_endpoint,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
                region_name="us-east-1",
            )
            for fname in ["movies.dat", "users.dat", "ratings.dat"]:
                s3.download_file(minio_bucket, f"movielens/latest/{fname}", str(raw_dir / fname))
            print("✅ Données téléchargées depuis MinIO")
        except Exception as e:
            print(f"⚠️  MinIO inaccessible ({e}) — utilisation des données locales")
            # En mode local : les fichiers sont montés dans le conteneur
            raw_dir = Path("/opt/airflow/data/raw")

        # Charger les données
        movies  = pd.read_csv(raw_dir / "movies.dat",  sep="::", engine="python",
                               encoding="latin-1", names=["MovieID","Title","Genres"])
        users   = pd.read_csv(raw_dir / "users.dat",   sep="::", engine="python",
                               encoding="latin-1", names=["UserID","Gender","Age","Occupation","Zip"])
        ratings = pd.read_csv(raw_dir / "ratings.dat", sep="::", engine="python",
                               encoding="latin-1", names=["UserID","MovieID","Rating","Timestamp"])

        # Feature engineering
        movies["Year"] = movies["Title"].str.extract(r"\((\d{4})\)").astype(float)
        genres_ohe     = movies["Genres"].str.get_dummies(sep="|")
        users["Gender_enc"] = (users["Gender"] == "M").astype(int)
        ratings["Liked"]    = (ratings["Rating"] >= 4).astype(int)

        user_stats  = ratings.groupby("UserID").agg(
            user_avg_rating=("Rating","mean"), user_n_ratings=("Rating","count"),
            user_std_rating=("Rating","std"),
        ).reset_index().fillna(0)
        movie_stats = ratings.groupby("MovieID").agg(
            movie_avg_rating=("Rating","mean"), movie_n_ratings=("Rating","count"),
        ).reset_index()

        movie_feats = pd.concat([movies[["MovieID","Year"]], genres_ohe], axis=1)
        user_feats  = users[["UserID","Gender_enc","Age","Occupation"]]

        df = (ratings[["UserID","MovieID","Liked"]]
              .merge(user_feats, on="UserID", how="left")
              .merge(movie_feats, on="MovieID", how="left")
              .merge(user_stats, on="UserID", how="left")
              .merge(movie_stats, on="MovieID", how="left")
              .fillna(0))

        genre_cols = list(genres_ohe.columns)
        stat_feats = ["user_avg_rating","user_n_ratings","user_std_rating",
                      "movie_avg_rating","movie_n_ratings"]
        base_feats = ["Gender_enc","Age","Occupation","Year"] + genre_cols
        feat_cols  = base_feats + stat_feats

        X = df[feat_cols]
        y = df["Liked"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Sauvegarder les artefacts
        X_train.to_csv(x_train_out.path, index=False)
        X_test .to_csv(x_test_out.path,  index=False)
        y_train.to_frame().to_csv(y_train_out.path, index=False)
        y_test .to_frame().to_csv(y_test_out.path,  index=False)

        with open(feature_cols_out.path, "w") as f:
            json.dump(feat_cols, f)

        print(f"✅ Données préparées : {len(X_train)} train | {len(X_test)} test | {len(feat_cols)} features")

    KFP_AVAILABLE = True

except ImportError:
    KFP_AVAILABLE = False
    prepare_data_component = None


# ── Implémentation locale (sans KFP) ──────────────────────────────────────

def prepare_data_local(
    test_size:    float = 0.2,
    random_state: int   = 42,
    output_dir:   str   = "data/processed",
) -> dict:
    """
    Équivalent local de prepare_data_component.
    Appelé par LocalPipelineRunner en mode développement.
    """
    import sys, json
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from training.data_loader import prepare_and_save

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result = prepare_and_save(test_size=test_size, random_state=random_state)

    print(f"[prepare_data] {result['n_train']} train | {result['n_test']} test | {result['n_features']} features")
    return {
        "x_train":      f"{output_dir}/X_train.csv",
        "x_test":       f"{output_dir}/X_test.csv",
        "y_train":      f"{output_dir}/y_train.csv",
        "y_test":       f"{output_dir}/y_test.csv",
        "feature_cols": f"{output_dir}/feature_columns.json",
        "stats":        result,
    }
