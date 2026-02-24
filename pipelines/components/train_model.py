"""
pipelines/components/train_model.py
Composant Kubeflow Pipelines — Entraînement du modèle Naïve Bayes.

Reçoit les artefacts Dataset de prepare_data_component,
entraîne le modèle, loggue dans MLflow, sauvegarde le modèle.
"""
from __future__ import annotations

try:
    from kfp import dsl
    from kfp.dsl import Dataset, Input, Model, Output, Metrics, component

    @component(
        base_image="python:3.10-slim",
        packages_to_install=[
            "scikit-learn==1.3.2",
            "pandas==2.1.4",
            "numpy==1.26.3",
            "mlflow==2.9.2",
            "boto3==1.28.85",
        ],
    )
    def train_model_component(
        # Hyperparamètres
        gaussian_var_smoothing: float,
        bernoulli_alpha:        float,
        gaussian_weight:        float,
        bernoulli_weight:       float,
        mlflow_tracking_uri:    str,
        experiment_name:        str,
        # Entrées
        x_train_in:       Input[Dataset],
        y_train_in:       Input[Dataset],
        feature_cols_in:  Input[Dataset],
        # Sorties
        model_output:     Output[Model],
        metrics_output:   Output[Metrics],
    ):
        """Composant KFP : entraîne et loggue le modèle Naïve Bayes."""
        import sys, os, json, pickle
        import pandas as pd
        import numpy as np
        from sklearn.naive_bayes   import GaussianNB, BernoulliNB
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        from sklearn.base import BaseEstimator, ClassifierMixin

        # Charger les données
        X_train = pd.read_csv(x_train_in.path)
        y_train = pd.read_csv(y_train_in.path).squeeze()
        with open(feature_cols_in.path) as f:
            feat_cols = json.load(f)

        X_train = X_train[feat_cols]

        # Charger le modèle depuis le code source
        # (en prod, l'image Docker contient le code du projet)
        sys.path.insert(0, "/opt/project")
        try:
            from training.models.naive_bayes_model import MovieNaiveBayesRecommender
        except ImportError:
            # Inline si le code n'est pas monté
            class MovieNaiveBayesRecommender(BaseEstimator, ClassifierMixin):
                CONTINUOUS = ["Gender_enc","Age","Occupation","Year",
                              "user_avg_rating","user_n_ratings","user_std_rating",
                              "movie_avg_rating","movie_n_ratings"]
                def __init__(self, gaussian_var_smoothing=1e-2, bernoulli_alpha=1.0,
                             gaussian_weight=0.6, bernoulli_weight=0.4):
                    self.gaussian_var_smoothing = gaussian_var_smoothing
                    self.bernoulli_alpha        = bernoulli_alpha
                    self.gaussian_weight        = gaussian_weight
                    self.bernoulli_weight       = bernoulli_weight
                def fit(self, X, y):
                    cont   = [c for c in self.CONTINUOUS if c in X.columns]
                    genres = [c for c in X.columns if c not in self.CONTINUOUS]
                    self.cont_cols_ = cont; self.genre_cols_ = genres
                    self.scaler_ = MinMaxScaler()
                    Xs = self.scaler_.fit_transform(X[cont])
                    self.gnb_ = GaussianNB(var_smoothing=self.gaussian_var_smoothing).fit(Xs, y)
                    if genres:
                        self.bnb_ = BernoulliNB(alpha=self.bernoulli_alpha).fit(X[genres].values, y)
                        self.has_genre_ = True
                    else:
                        self.has_genre_ = False
                    self.classes_ = np.array([0, 1])
                    return self
                def predict_proba(self, X):
                    Xs = self.scaler_.transform(X[self.cont_cols_])
                    pg = self.gnb_.predict_proba(Xs)
                    if self.has_genre_:
                        pb = self.bnb_.predict_proba(X[self.genre_cols_].values)
                        t  = self.gaussian_weight + self.bernoulli_weight
                        return (self.gaussian_weight * pg + self.bernoulli_weight * pb) / t
                    return pg
                def predict(self, X):
                    return (self.predict_proba(X)[:,1] >= 0.5).astype(int)

        # Entraîner
        model = MovieNaiveBayesRecommender(
            gaussian_var_smoothing=gaussian_var_smoothing,
            bernoulli_alpha=bernoulli_alpha,
            gaussian_weight=gaussian_weight,
            bernoulli_weight=bernoulli_weight,
        )
        model.fit(X_train, y_train)

        # Cross-validation
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        cv_mean, cv_std = float(cv.mean()), float(cv.std())

        # Métriques sur le train set (le test est dans evaluate_component)
        y_pred  = model.predict(X_train)
        y_proba = model.predict_proba(X_train)
        train_acc = float(accuracy_score(y_train, y_pred))
        train_auc = float(roc_auc_score(y_train, y_proba[:,1]))

        # Logger dans MLflow
        try:
            import mlflow, mlflow.sklearn
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name="kubeflow_component"):
                mlflow.log_params({
                    "gaussian_var_smoothing": gaussian_var_smoothing,
                    "bernoulli_alpha":        bernoulli_alpha,
                    "gaussian_weight":        gaussian_weight,
                    "bernoulli_weight":       bernoulli_weight,
                    "n_train":                len(X_train),
                    "n_features":             len(feat_cols),
                })
                mlflow.log_metrics({
                    "train_accuracy": train_acc,
                    "train_roc_auc":  train_auc,
                    "cv_accuracy_mean": cv_mean,
                    "cv_accuracy_std":  cv_std,
                })
                mlflow.sklearn.log_model(model, "model",
                                         registered_model_name="naive-bayes-recommender")
            print(f"✅ Run MLflow loggué")
        except Exception as e:
            print(f"⚠️  MLflow indisponible : {e}")

        # Sauvegarder le modèle
        os.makedirs(model_output.path, exist_ok=True)
        with open(os.path.join(model_output.path, "model.pkl"), "wb") as f:
            pickle.dump({"model": model, "feature_columns": feat_cols}, f)

        # Métriques KFP
        metrics_output.log_metric("cv_accuracy_mean", cv_mean)
        metrics_output.log_metric("cv_accuracy_std",  cv_std)
        metrics_output.log_metric("train_accuracy",   train_acc)

        print(f"✅ Modèle entraîné — CV: {cv_mean:.4f} ± {cv_std:.4f}")

    KFP_AVAILABLE = True

except ImportError:
    KFP_AVAILABLE = False
    train_model_component = None


# ── Implémentation locale ──────────────────────────────────────────────────

def train_model_local(
    gaussian_var_smoothing: float = 1e-2,
    bernoulli_alpha:        float = 1.0,
    gaussian_weight:        float = 0.6,
    bernoulli_weight:       float = 0.4,
    data_dir:    str = "data/processed",
    output_dir:  str = "models",
    mlflow_uri:  str = None,
) -> dict:
    """Équivalent local de train_model_component."""
    import sys, json, pickle
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from training.models.naive_bayes_model import MovieNaiveBayesRecommender
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score, roc_auc_score
    import pandas as pd

    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").squeeze()
    with open(f"{data_dir}/feature_columns.json") as f:
        feat_cols = json.load(f)

    model = MovieNaiveBayesRecommender(
        gaussian_var_smoothing=gaussian_var_smoothing,
        bernoulli_alpha=bernoulli_alpha,
        gaussian_weight=gaussian_weight,
        bernoulli_weight=bernoulli_weight,
    )
    model.fit(X_train[feat_cols], y_train)

    cv      = cross_val_score(model, X_train[feat_cols], y_train, cv=5, scoring="accuracy")
    y_pred  = model.predict(X_train[feat_cols])
    y_proba = model.predict_proba(X_train[feat_cols])

    metrics = {
        "cv_accuracy_mean": round(float(cv.mean()), 4),
        "cv_accuracy_std":  round(float(cv.std()),  4),
        "train_accuracy":   round(float(accuracy_score(y_train, y_pred)), 4),
        "train_roc_auc":    round(float(roc_auc_score(y_train, y_proba[:,1])), 4),
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    bundle = {"model": model, "feature_columns": feat_cols, "metrics": metrics}
    model_path = f"{output_dir}/kfp_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"[train_model] CV={metrics['cv_accuracy_mean']:.4f}±{metrics['cv_accuracy_std']:.4f}")
    return {"model_path": model_path, "metrics": metrics}
