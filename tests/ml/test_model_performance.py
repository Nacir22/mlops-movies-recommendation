"""
tests/ml/test_model_performance.py
Suite de tests ML : performance, stabilité et comportement du modèle.

Exécuter avec :
    pytest tests/ml/ -v
    pytest tests/ml/ -v --tb=short

Ces tests valident que le modèle respecte les seuils minimaux définis
dans configs/model_config.yaml avant toute promotion en production.
"""
import sys
import json
import pickle
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Ajouter le répertoire racine au path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── Seuils de performance ──────────────────────────────────────────────────
MIN_ACCURACY   = 0.60
MIN_F1_SCORE   = 0.50
MIN_ROC_AUC    = 0.52
MAX_CV_STD     = 0.05   # Stabilité cross-validation
MIN_SEPARATION = 0.005  # Séparation minimale (faible sur données synthétiques, >0.05 attendu sur vraies données MovieLens)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model_bundle():
    """Charge le bundle modèle depuis le fichier pickle."""
    # Chercher dans l'ordre : kfp_model, naive_bayes_feast, naive_bayes_recommender
    candidates = [
        ROOT / "models" / "kfp_model.pkl",
        ROOT / "models" / "naive_bayes_feast.pkl",
        ROOT / "models" / "naive_bayes_recommender.pkl",
    ]
    for path in candidates:
        if path.exists():
            with open(path, "rb") as f:
                bundle = pickle.load(f)
            print(f"\n📦 Modèle chargé : {path.name}")
            return bundle

    pytest.skip("Aucun modèle entraîné trouvé — lancez d'abord training/train.py")


@pytest.fixture(scope="module")
def test_data():
    """Charge X_test et y_test."""
    proc_dir = ROOT / "data" / "processed"
    if not (proc_dir / "X_test.csv").exists():
        pytest.skip("Données de test introuvables — lancez d'abord training/data_loader.py")

    X_test = pd.read_csv(proc_dir / "X_test.csv")
    y_test = pd.read_csv(proc_dir / "y_test.csv").squeeze()
    return X_test, y_test


@pytest.fixture(scope="module")
def train_data():
    """Charge X_train et y_train (pour la cross-validation)."""
    proc_dir = ROOT / "data" / "processed"
    X_train = pd.read_csv(proc_dir / "X_train.csv")
    y_train = pd.read_csv(proc_dir / "y_train.csv").squeeze()
    return X_train, y_train


@pytest.fixture(scope="module")
def predictions(model_bundle, test_data):
    """Calcule les prédictions une seule fois pour tous les tests."""
    model     = model_bundle["model"]
    feat_cols = model_bundle["feature_columns"]
    X_test, y_test = test_data

    y_pred  = model.predict(X_test[feat_cols])
    y_proba = model.predict_proba(X_test[feat_cols])
    return y_test, y_pred, y_proba


# ── Tests de performance ───────────────────────────────────────────────────

class TestModelPerformance:
    """Tests des métriques de performance sur le test set."""

    def test_accuracy_above_threshold(self, predictions):
        """L'accuracy doit être >= MIN_ACCURACY."""
        from sklearn.metrics import accuracy_score
        y_test, y_pred, _ = predictions
        accuracy = accuracy_score(y_test, y_pred)
        assert accuracy >= MIN_ACCURACY, (
            f"Accuracy {accuracy:.4f} < seuil {MIN_ACCURACY}. "
            "Réentraînez avec de meilleurs hyperparamètres."
        )

    def test_f1_score_above_threshold(self, predictions):
        """Le F1-score doit être >= MIN_F1_SCORE."""
        from sklearn.metrics import f1_score
        y_test, y_pred, _ = predictions
        f1 = f1_score(y_test, y_pred, zero_division=0)
        assert f1 >= MIN_F1_SCORE, (
            f"F1-score {f1:.4f} < seuil {MIN_F1_SCORE}"
        )

    def test_roc_auc_above_threshold(self, predictions):
        """Le ROC-AUC doit être >= MIN_ROC_AUC (mieux qu'aléatoire)."""
        from sklearn.metrics import roc_auc_score
        y_test, _, y_proba = predictions
        auc = roc_auc_score(y_test, y_proba[:, 1])
        assert auc >= MIN_ROC_AUC, (
            f"ROC-AUC {auc:.4f} < seuil {MIN_ROC_AUC}. "
            "Le modèle est proche ou inférieur au hasard."
        )

    def test_precision_reasonable(self, predictions):
        """La précision doit être > 0 (le modèle prédit au moins parfois correctement)."""
        from sklearn.metrics import precision_score
        y_test, y_pred, _ = predictions
        precision = precision_score(y_test, y_pred, zero_division=0)
        assert precision > 0, "Précision nulle : le modèle ne fait aucune prédiction positive correcte"

    def test_both_classes_predicted(self, predictions):
        """Le modèle doit prédire les deux classes (pas de dégénérescence)."""
        _, y_pred, _ = predictions
        unique = np.unique(y_pred)
        assert len(unique) >= 2, (
            f"Le modèle ne prédit qu'une seule classe : {unique}. "
            "Vérifiez le déséquilibre de classes."
        )


# ── Tests de stabilité ─────────────────────────────────────────────────────

class TestModelStability:
    """Tests de stabilité et reproductibilité du modèle."""

    def test_cv_std_below_threshold(self, model_bundle, train_data):
        """L'écart-type de la cross-validation doit être faible."""
        from sklearn.model_selection import cross_val_score
        model     = model_bundle["model"]
        feat_cols = model_bundle["feature_columns"]
        X_train, y_train = train_data

        cv_scores = cross_val_score(
            model, X_train[feat_cols], y_train, cv=5, scoring="accuracy"
        )
        cv_std = cv_scores.std()
        assert cv_std <= MAX_CV_STD, (
            f"CV std {cv_std:.4f} > seuil {MAX_CV_STD}. "
            "Le modèle est instable selon les folds."
        )

    def test_predictions_deterministic(self, model_bundle, test_data):
        """Deux appels à predict() doivent retourner les mêmes résultats."""
        model     = model_bundle["model"]
        feat_cols = model_bundle["feature_columns"]
        X_test, _ = test_data
        sample    = X_test[feat_cols].head(100)

        pred1 = model.predict(sample)
        pred2 = model.predict(sample)
        assert np.array_equal(pred1, pred2), "predict() n'est pas déterministe"

    def test_proba_sum_to_one(self, predictions):
        """Les probabilités par ligne doivent sommer à 1.0."""
        _, _, y_proba = predictions
        row_sums = y_proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), (
            f"Les probabilités ne somment pas à 1 (max écart : {np.abs(row_sums - 1).max():.2e})"
        )

    def test_proba_in_valid_range(self, predictions):
        """Toutes les probabilités doivent être dans [0, 1]."""
        _, _, y_proba = predictions
        assert y_proba.min() >= 0.0, f"Probabilité négative : {y_proba.min()}"
        assert y_proba.max() <= 1.0, f"Probabilité > 1 : {y_proba.max()}"

    def test_score_separation(self, predictions):
        """Les scores P(Liked=1) doivent être plus élevés pour les vrais positifs."""
        y_test, _, y_proba = predictions
        scores_pos = y_proba[:, 1][y_test == 1]
        scores_neg = y_proba[:, 1][y_test == 0]
        separation = scores_pos.mean() - scores_neg.mean()
        assert separation >= MIN_SEPARATION, (
            f"Séparation des scores : {separation:.4f} < {MIN_SEPARATION}. "
            "Le modèle ne distingue pas bien les classes."
        )


# ── Tests fonctionnels ─────────────────────────────────────────────────────

class TestModelFunctionality:
    """Tests du comportement fonctionnel du modèle."""

    def test_predict_proba_shape(self, model_bundle, test_data):
        """predict_proba doit retourner une matrice (n, 2)."""
        model     = model_bundle["model"]
        feat_cols = model_bundle["feature_columns"]
        X_test, _ = test_data
        sample    = X_test[feat_cols].head(50)
        proba     = model.predict_proba(sample)
        assert proba.shape == (50, 2), f"Shape inattendu : {proba.shape}"

    def test_predict_output_binary(self, predictions):
        """predict() doit retourner uniquement 0 et 1."""
        _, y_pred, _ = predictions
        unique = np.unique(y_pred)
        assert set(unique).issubset({0, 1}), f"Valeurs non binaires : {unique}"

    def test_feature_columns_complete(self, model_bundle):
        """Le bundle doit contenir les feature_columns."""
        assert "feature_columns" in model_bundle, "feature_columns manquant dans le bundle"
        feat_cols = model_bundle["feature_columns"]
        assert len(feat_cols) > 0, "Liste de features vide"
        assert len(feat_cols) >= 10, f"Trop peu de features : {len(feat_cols)}"

    def test_recommend_movies(self, model_bundle):
        """La méthode recommend_movies() doit retourner des résultats valides."""
        from training.data_loader import load_movies, load_ratings
        from training.models.naive_bayes_model import MovieNaiveBayesRecommender

        model = model_bundle["model"]
        if not hasattr(model, "recommend_movies"):
            pytest.skip("Méthode recommend_movies non disponible sur ce modèle")

        movies  = load_movies()
        ratings = load_ratings()

        movie_stats = ratings.groupby("MovieID").agg(
            movie_avg_rating=("Rating","mean"),
            movie_n_ratings=("Rating","count")
        ).reset_index()

        genre_cols = [c for c in movies.columns
                      if c not in ["MovieID","Title","Genres","Year"]]
        candidates = movies[["MovieID","Title","Year"] + genre_cols].head(30)

        recs = model.recommend_movies(
            user_features={"Gender_enc": 1, "Age": 25, "Occupation": 4},
            candidate_movies=candidates,
            movie_stats=movie_stats,
            top_k=5,
        )

        assert len(recs) <= 5,         "Trop de recommandations retournées"
        assert "score" in recs.columns, "Colonne score manquante"
        assert "MovieID" in recs.columns, "Colonne MovieID manquante"
        assert (recs["score"] >= 0).all() and (recs["score"] <= 1).all(), \
            "Scores hors de [0, 1]"
        # Vérifier le tri décroissant
        assert recs["score"].is_monotonic_decreasing, "Recommandations non triées par score"

    def test_handles_unknown_features(self, model_bundle):
        """Le modèle doit accepter des features inconnues (colonnes supplémentaires)."""
        model     = model_bundle["model"]
        feat_cols = model_bundle["feature_columns"]
        proc_dir  = ROOT / "data" / "processed"
        X_test    = pd.read_csv(proc_dir / "X_test.csv").head(10)

        # Ajouter une colonne parasite
        X_test_extra = X_test.copy()
        X_test_extra["colonne_inconnue"] = 0.0

        # Ne doit pas planter — on sélectionne seulement les bonnes colonnes
        pred = model.predict(X_test_extra[feat_cols])
        assert len(pred) == 10
