"""
training/models/naive_bayes_model.py
Modèle Naïve Bayes hybride pour la recommandation de films MovieLens.

Architecture :
  ┌─────────────────────────────────────────────────────────┐
  │  Features continues         →  GaussianNB               │
  │  (Age, Year, stats…)                                    │
  │                               ↘                         │
  │                                 Moyenne pondérée → P(y) │
  │                               ↗                         │
  │  Features binaires          →  BernoulliNB              │
  │  (genres one-hot)                                       │
  └─────────────────────────────────────────────────────────┘
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


# Features continues (encodées numériquement)
CONTINUOUS_FEATURES = ["Gender_enc", "Age", "Occupation", "Year",
                        "user_avg_rating", "user_n_ratings", "user_std_rating",
                        "movie_avg_rating", "movie_n_ratings"]


class MovieNaiveBayesRecommender(BaseEstimator, ClassifierMixin):
    """
    Recommandeur hybride Naïve Bayes.

    Combine :
    - GaussianNB sur les features continues (normalisées)
    - BernoulliNB sur les genres (one-hot binaires)
    via une moyenne pondérée des probabilités a posteriori.

    Paramètres
    ----------
    gaussian_var_smoothing : float
        Lissage de variance pour GaussianNB (défaut 1e-2).
    bernoulli_alpha : float
        Lissage de Laplace pour BernoulliNB (défaut 1.0).
    gaussian_weight : float
        Poids des features continues dans la fusion (défaut 0.6).
    bernoulli_weight : float
        Poids des features binaires dans la fusion (défaut 0.4).
    """

    def __init__(
        self,
        gaussian_var_smoothing: float = 1e-2,
        bernoulli_alpha:        float = 1.0,
        gaussian_weight:        float = 0.6,
        bernoulli_weight:       float = 0.4,
    ):
        self.gaussian_var_smoothing = gaussian_var_smoothing
        self.bernoulli_alpha        = bernoulli_alpha
        self.gaussian_weight        = gaussian_weight
        self.bernoulli_weight       = bernoulli_weight

    # ── Helpers ────────────────────────────────────────────────────────────

    def _split_features(self, X: pd.DataFrame):
        """Sépare X en features continues et binaires (genres)."""
        cont_cols  = [c for c in CONTINUOUS_FEATURES if c in X.columns]
        genre_cols = [c for c in X.columns if c not in CONTINUOUS_FEATURES]
        return X[cont_cols], X[genre_cols], cont_cols, genre_cols

    # ── API scikit-learn ───────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y):
        X_cont, X_genre, cont_cols, genre_cols = self._split_features(X)

        self.cont_cols_  = cont_cols
        self.genre_cols_ = genre_cols
        self.classes_    = np.array([0, 1])

        # Normaliser les features continues dans [0, 1]
        self.scaler_ = MinMaxScaler()
        X_cont_s = self.scaler_.fit_transform(X_cont)

        # Entraîner les deux modèles
        self.gnb_ = GaussianNB(var_smoothing=self.gaussian_var_smoothing)
        self.bnb_ = BernoulliNB(alpha=self.bernoulli_alpha)

        self.gnb_.fit(X_cont_s, y)

        if X_genre.shape[1] > 0:
            self.bnb_.fit(X_genre.values.astype(float), y)
            self.has_genre_ = True
        else:
            self.has_genre_ = False

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_cont, X_genre, _, _ = self._split_features(X)

        # Features continues
        X_cont_s = self.scaler_.transform(X_cont)
        p_gnb    = self.gnb_.predict_proba(X_cont_s)          # (n, 2)

        if self.has_genre_ and X_genre.shape[1] > 0:
            # Features binaires
            p_bnb = self.bnb_.predict_proba(X_genre.values.astype(float))
            total = self.gaussian_weight + self.bernoulli_weight
            proba = (self.gaussian_weight * p_gnb +
                     self.bernoulli_weight * p_bnb) / total
        else:
            proba = p_gnb

        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X: pd.DataFrame, y) -> float:
        return accuracy_score(y, self.predict(X))

    # ── Méthode de recommandation ──────────────────────────────────────────

    def recommend_movies(
        self,
        user_features:    dict,
        candidate_movies: pd.DataFrame,
        user_stats:       pd.DataFrame = None,
        movie_stats:      pd.DataFrame = None,
        top_k:            int = 10,
    ) -> pd.DataFrame:
        """
        Recommande les top_k films pour un utilisateur.

        Paramètres
        ----------
        user_features : dict
            Profil utilisateur : {'Gender_enc': 1, 'Age': 25, 'Occupation': 4}
        candidate_movies : DataFrame
            Films candidats avec colonnes MovieID, Title, Year, genres...
        user_stats : DataFrame optionnel
            Contient user_avg_rating, user_n_ratings, user_std_rating
        movie_stats : DataFrame optionnel
            Contient movie_avg_rating, movie_n_ratings par MovieID
        top_k : int
            Nombre de recommandations à retourner

        Retourne
        --------
        DataFrame trié par score décroissant : MovieID, Title, score
        """
        n = len(candidate_movies)

        # Construire le DataFrame de features
        rows = []
        for _, movie_row in candidate_movies.iterrows():
            row = dict(user_features)

            # Features film
            row["Year"] = float(movie_row.get("Year", 0) or 0)
            for c in self.genre_cols_:
                row[c] = float(movie_row.get(c, 0) or 0)

            # Stats utilisateur (moyennes globales si non fournies)
            if user_stats is not None and "user_avg_rating" in self.cont_cols_:
                row["user_avg_rating"] = user_features.get(
                    "user_avg_rating", float(user_stats["user_avg_rating"].mean()))
                row["user_n_ratings"]  = user_features.get(
                    "user_n_ratings",  float(user_stats["user_n_ratings"].mean()))
                row["user_std_rating"] = user_features.get(
                    "user_std_rating", float(user_stats["user_std_rating"].mean()))

            # Stats film
            if movie_stats is not None and "movie_avg_rating" in self.cont_cols_:
                mid = movie_row["MovieID"]
                ms  = movie_stats[movie_stats["MovieID"] == mid]
                row["movie_avg_rating"] = float(ms["movie_avg_rating"].values[0]) if len(ms) else 3.5
                row["movie_n_ratings"]  = float(ms["movie_n_ratings"].values[0])  if len(ms) else 10.0

            rows.append(row)

        feat_df = pd.DataFrame(rows)

        # Aligner les colonnes sur celles vues à l'entraînement
        all_cols = self.cont_cols_ + self.genre_cols_
        for c in all_cols:
            if c not in feat_df.columns:
                feat_df[c] = 0.0
        feat_df = feat_df[all_cols].fillna(0)

        proba = self.predict_proba(feat_df)[:, 1]

        result = candidate_movies[["MovieID", "Title"]].copy().reset_index(drop=True)
        result["score"] = proba
        return (result
                .sort_values("score", ascending=False)
                .head(top_k)
                .reset_index(drop=True))
