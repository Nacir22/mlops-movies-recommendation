"""
serving/fastapi/model_loader.py
Chargement du modèle depuis MLflow Registry ou pickle local.

Priorité :
  1. MLflow Registry (stage=Production)  → production
  2. models/naive_bayes_feast.pkl         → entraîné avec Feast
  3. models/naive_bayes_recommender.pkl   → baseline

Expose aussi le catalogue de films et les stats
utilisées pour enrichir les requêtes en temps réel.
"""
from __future__ import annotations

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent


class ModelBundle:
    """
    Conteneur pour le modèle chargé, ses métadonnées et le catalogue.
    Chargé une seule fois au démarrage de l'application (lifespan).
    """

    def __init__(self):
        self.model           = None
        self.feature_columns: list[str] = []
        self.version:  str = "unknown"
        self.source:   str = "none"
        self.metrics:  dict = {}
        self.movies_df: Optional[pd.DataFrame] = None
        self.movie_stats: Optional[pd.DataFrame] = None
        self.user_stats:  Optional[pd.DataFrame] = None
        self._loaded = False

    def load(self) -> "ModelBundle":
        """Charge le modèle et le catalogue. Appelé au démarrage FastAPI."""
        self._load_model()
        self._load_catalogue()
        self._loaded = True
        log.info(
            "✅ ModelBundle chargé — version=%s source=%s features=%d",
            self.version, self.source, len(self.feature_columns),
        )
        return self

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None

    # ── Chargement du modèle ───────────────────────────────────────────────

    def _load_model(self):
        """Tente MLflow d'abord, puis pickle local."""
        # 1. Essayer MLflow Registry
        if self._try_mlflow():
            return
        # 2. Fallback : pickle local
        self._load_pickle()

    def _try_mlflow(self) -> bool:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        model_name   = os.getenv("MODEL_NAME", "naive-bayes-recommender")
        stage        = os.getenv("MODEL_STAGE", "Production")

        try:
            import mlflow.sklearn
            model_uri = f"models:/{model_name}/{stage}"
            self.model = mlflow.sklearn.load_model(model_uri)

            # Récupérer la version et les métriques
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            client   = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(model_name, stages=[stage])
            if versions:
                v = versions[0]
                run = client.get_run(v.run_id)
                self.version  = f"v{v.version}"
                self.metrics  = run.data.metrics
                # Récupérer les feature_columns depuis les artefacts
                fc_path = client.download_artifacts(v.run_id, "feature_columns.json")
                with open(fc_path) as f:
                    self.feature_columns = json.load(f)
            self.source = "mlflow"
            log.info("Modèle chargé depuis MLflow Registry (%s/%s)", model_name, stage)
            return True

        except Exception as e:
            log.debug("MLflow indisponible : %s", e)
            return False

    def _load_pickle(self):
        """Charge depuis les fichiers pickle locaux (ordre de préférence)."""
        candidates = [
            ROOT / "models" / "kfp_model.pkl",
            ROOT / "models" / "naive_bayes_feast.pkl",
            ROOT / "models" / "naive_bayes_recommender.pkl",
        ]
        for path in candidates:
            if path.exists():
                with open(path, "rb") as f:
                    bundle = pickle.load(f)
                self.model           = bundle["model"]
                self.feature_columns = bundle.get("feature_columns", [])
                self.metrics         = bundle.get("metrics", {})
                self.version         = bundle.get("run_name", path.stem)
                self.source          = f"pickle:{path.name}"
                log.info("Modèle chargé depuis pickle : %s", path.name)
                return

        raise RuntimeError(
            "Aucun modèle disponible. "
            "Lancez d'abord : python training/train.py"
        )

    # ── Chargement du catalogue ────────────────────────────────────────────

    def _load_catalogue(self):
        """Charge les films, stats films et stats utilisateurs."""
        try:
            import sys
            sys.path.insert(0, str(ROOT))
            from training.data_loader import load_movies, load_ratings

            movies  = load_movies()
            ratings = load_ratings()

            # Catalogue complet
            self.movies_df = movies

            # Stats par film
            self.movie_stats = (
                ratings.groupby("MovieID")
                .agg(movie_avg_rating=("Rating","mean"),
                     movie_n_ratings =("Rating","count"))
                .reset_index()
            )

            # Stats globales utilisateurs (moyennes pour cold-start)
            self.user_stats = (
                ratings.groupby("UserID")
                .agg(user_avg_rating =("Rating","mean"),
                     user_n_ratings  =("Rating","count"),
                     user_std_rating =("Rating","std"))
                .reset_index().fillna(0)
            )

            log.info(
                "Catalogue chargé : %d films | movie_stats : %d",
                len(movies), len(self.movie_stats),
            )
        except Exception as e:
            log.warning("Catalogue non disponible : %s", e)
            self.movies_df   = pd.DataFrame()
            self.movie_stats = pd.DataFrame()
            self.user_stats  = pd.DataFrame()

    # ── API de prédiction ──────────────────────────────────────────────────

    def predict_like(self, user_features: dict, movie_id: int) -> tuple[bool, float]:
        """
        Prédit si un utilisateur aimera un film.
        Retourne (will_like: bool, probability: float).
        """
        movie_row = self._get_movie_row(movie_id)
        feat_row  = self._build_feature_row(user_features, movie_row)
        proba     = self.model.predict_proba(feat_row)[0, 1]
        return bool(proba >= 0.5), float(proba)

    def recommend(
        self,
        user_features:    dict,
        top_k:            int        = 10,
        exclude_movie_ids: list[int] = None,
    ) -> pd.DataFrame:
        """
        Retourne les top_k films recommandés triés par score décroissant.
        """
        if self.movies_df is None or len(self.movies_df) == 0:
            return pd.DataFrame()

        exclude = set(exclude_movie_ids or [])
        candidates = self.movies_df[~self.movies_df["MovieID"].isin(exclude)].copy()

        # Construire la matrice de features pour tous les candidats
        rows = []
        for _, movie_row in candidates.iterrows():
            rows.append(self._build_feature_row(user_features, movie_row, as_dict=True))

        feat_df = pd.DataFrame(rows)
        for c in self.feature_columns:
            if c not in feat_df.columns:
                feat_df[c] = 0.0
        feat_df = feat_df[self.feature_columns].fillna(0)

        probas = self.model.predict_proba(feat_df)[:, 1]

        result = candidates[["MovieID","Title","Genres"]].copy().reset_index(drop=True)
        result["score"] = probas
        return result.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)

    def _get_movie_row(self, movie_id: int) -> pd.Series:
        """Récupère la ligne d'un film par son ID."""
        if self.movies_df is not None and len(self.movies_df) > 0:
            rows = self.movies_df[self.movies_df["MovieID"] == movie_id]
            if len(rows) > 0:
                return rows.iloc[0]
        return pd.Series({"MovieID": movie_id, "Year": 2000.0})

    def _build_feature_row(
        self,
        user_features: dict,
        movie_row:     pd.Series,
        as_dict:       bool = False,
    ):
        """Construit une ligne de features à partir du profil utilisateur + film."""
        row = dict(user_features)

        # Features film
        row["Year"] = float(movie_row.get("Year", 0) or 0)

        # Genres one-hot
        genre_cols = [c for c in self.feature_columns
                      if c not in {"Gender_enc","Age","Occupation","Year",
                                   "user_avg_rating","user_n_ratings","user_std_rating",
                                   "movie_avg_rating","movie_n_ratings"}]
        for c in genre_cols:
            row[c] = float(movie_row.get(c, 0) or 0)

        # Stats film depuis movie_stats
        if self.movie_stats is not None and len(self.movie_stats) > 0:
            mid  = int(movie_row.get("MovieID", 0))
            ms   = self.movie_stats[self.movie_stats["MovieID"] == mid]
            row["movie_avg_rating"] = float(ms["movie_avg_rating"].values[0]) if len(ms) else 3.5
            row["movie_n_ratings"]  = float(ms["movie_n_ratings"].values[0])  if len(ms) else 10.0
        else:
            row["movie_avg_rating"] = 3.5
            row["movie_n_ratings"]  = 10.0

        if as_dict:
            return row

        feat_df = pd.DataFrame([row])
        for c in self.feature_columns:
            if c not in feat_df.columns:
                feat_df[c] = 0.0
        return feat_df[self.feature_columns].fillna(0)


# Singleton global chargé au démarrage
_bundle: Optional[ModelBundle] = None


def get_bundle() -> ModelBundle:
    """Retourne le bundle chargé (dependency injection FastAPI)."""
    global _bundle
    if _bundle is None or not _bundle.is_loaded:
        _bundle = ModelBundle().load()
    return _bundle


def load_bundle() -> ModelBundle:
    """Charge et retourne le bundle (appelé dans lifespan)."""
    global _bundle
    _bundle = ModelBundle().load()
    return _bundle
