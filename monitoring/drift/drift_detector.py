"""
monitoring/drift/drift_detector.py
Moteur de détection de data drift pour MovieLens MLOps.

Implémente les métriques de drift sans dépendance externe :
  - PSI  (Population Stability Index)       : drift global des distributions
  - KS   (Kolmogorov-Smirnov test)          : drift univarié features continues
  - Chi2 (Chi-squared test)                 : drift univarié features catégorielles
  - Prédiction drift                        : évolution de la distribution des scores

Seuils de décision :
  PSI  < 0.10  → pas de drift (vert)
  PSI  0.10–0.25 → drift modéré (orange)
  PSI  > 0.25  → drift significatif (rouge) → déclencher ré-entraînement

Compatible Evidently : la classe DriftReport expose une API identique
à evidently.report.Report pour faciliter la migration.
"""
from __future__ import annotations

import json
import math
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
from scipy import stats

log = logging.getLogger(__name__)

# ── Constantes ──────────────────────────────────────────────────────────────

PSI_NO_DRIFT  = 0.10   # PSI < 0.10 → pas de drift
PSI_MODERATE  = 0.25   # PSI 0.10–0.25 → drift modéré
KS_ALPHA      = 0.05   # Seuil de significativité pour le test KS
CHI2_ALPHA    = 0.05   # Seuil de significativité pour le chi2

REPORTS_DIR = Path(__file__).parent.parent / "evidently" / "reports"


# ── Fonctions de calcul ──────────────────────────────────────────────────────

def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    Calcule le Population Stability Index entre deux distributions.

    PSI = Σ (P_current - P_reference) × ln(P_current / P_reference)

    Interprétation :
      PSI < 0.10  → distribution stable
      PSI 0.10–0.25 → changement modéré, surveiller
      PSI > 0.25  → changement significatif, ré-entraîner
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current,   dtype=float)

    # Créer les bins sur la plage de référence
    global_min = min(ref.min(), cur.min())
    global_max = max(ref.max(), cur.max())
    if global_max == global_min:
        return 0.0   # Distribution constante → pas de drift

    bins = np.linspace(global_min, global_max, n_bins + 1)
    bins[-1] += 1e-10  # inclure la valeur max

    ref_counts, _ = np.histogram(ref, bins=bins)
    cur_counts, _ = np.histogram(cur, bins=bins)

    # Éviter les divisions par zéro (remplacer 0 par 0.001)
    ref_pct = np.where(ref_counts == 0, 0.001, ref_counts / len(ref))
    cur_pct = np.where(cur_counts == 0, 0.001, cur_counts / len(cur))

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(abs(psi), 6)


def compute_ks_test(
    reference: np.ndarray, current: np.ndarray
) -> tuple[float, float, bool]:
    """
    Kolmogorov-Smirnov test : détecte un drift dans les distributions continues.

    Returns:
        ks_stat   : statistique KS ∈ [0, 1]
        p_value   : p-value du test
        drifted   : True si p_value < KS_ALPHA (drift détecté)
    """
    ks_stat, p_value = stats.ks_2samp(reference, current)
    return float(ks_stat), float(p_value), bool(p_value < KS_ALPHA)


def compute_chi2_test(
    reference: np.ndarray, current: np.ndarray
) -> tuple[float, float, bool]:
    """
    Chi-2 test : détecte un drift dans les distributions catégorielles/binaires.

    Returns:
        chi2_stat : statistique chi2
        p_value   : p-value du test
        drifted   : True si p_value < CHI2_ALPHA
    """
    categories = np.union1d(np.unique(reference), np.unique(current))
    ref_counts  = np.array([(reference == c).sum() for c in categories], dtype=float)
    cur_counts  = np.array([(current   == c).sum() for c in categories], dtype=float)

    # Normaliser pour avoir les fréquences attendues
    ref_freq = ref_counts / ref_counts.sum()
    expected = ref_freq * cur_counts.sum()

    # Exclure les catégories avec expected=0
    mask     = expected > 0
    if mask.sum() < 2:
        return 0.0, 1.0, False

    chi2_stat, p_value = stats.chisquare(cur_counts[mask], f_exp=expected[mask])
    return float(chi2_stat), float(p_value), bool(p_value < CHI2_ALPHA)


def compute_wasserstein(reference: np.ndarray, current: np.ndarray) -> float:
    """Distance de Wasserstein (Earth Mover's Distance) entre deux distributions."""
    return float(stats.wasserstein_distance(reference, current))


# ── Classe principale ────────────────────────────────────────────────────────

class DriftDetector:
    """
    Détecteur de drift pour les features MovieLens.

    Utilise les données de référence (train set P1) pour détecter
    si les nouvelles données de production ont drifté.

    API compatible Evidently :
      detector = DriftDetector(reference_data, current_data)
      report   = detector.run()
      report.save_json("reports/drift_report_20240115.json")
      report.save_html("reports/drift_report_20240115.html")
    """

    # Features continues → KS + PSI
    CONTINUOUS_FEATURES = [
        "Age", "Occupation", "Year",
        "user_avg_rating", "user_n_ratings", "user_std_rating",
        "movie_avg_rating", "movie_n_ratings",
    ]
    # Features binaires → Chi2 + PSI
    BINARY_FEATURES = ["Gender_enc"]
    # Genres → Chi2
    GENRE_FEATURES = [
        "Action","Adventure","Animation","Comedy","Crime","Documentary",
        "Drama","Fantasy","Horror","Musical","Mystery","Romance",
        "Sci-Fi","Thriller","War","Western",
    ]

    def __init__(
        self,
        reference_data: pd.DataFrame,
        current_data:   pd.DataFrame,
        feature_columns: Optional[list[str]] = None,
        dataset_name:    str = "movielens",
    ):
        self.reference      = reference_data
        self.current        = current_data
        self.feature_columns = feature_columns or self._detect_columns()
        self.dataset_name   = dataset_name
        self._results: Optional[dict] = None

    def _detect_columns(self) -> list[str]:
        """Identifie les colonnes communes aux deux datasets."""
        return [c for c in self.reference.columns
                if c in self.current.columns
                and c not in {"UserID","MovieID","Liked","Rating","Timestamp"}]

    def run(self) -> "DriftReport":
        """Lance l'analyse de drift et retourne un DriftReport."""
        results = {
            "dataset":    self.dataset_name,
            "timestamp":  datetime.utcnow().isoformat(),
            "n_reference": len(self.reference),
            "n_current":   len(self.current),
            "features":   {},
            "summary":    {},
        }

        drifted_features = []

        for feature in self.feature_columns:
            if feature not in self.reference.columns:
                continue
            if feature not in self.current.columns:
                continue

            ref_vals = self.reference[feature].dropna().values
            cur_vals = self.current[feature].dropna().values

            if len(ref_vals) < 10 or len(cur_vals) < 10:
                continue

            feat_result = self._analyze_feature(feature, ref_vals, cur_vals)
            results["features"][feature] = feat_result

            if feat_result.get("drifted", False):
                drifted_features.append(feature)

        # Score global : moyenne des PSI
        psi_values = [
            v["psi"] for v in results["features"].values()
            if "psi" in v
        ]
        global_psi = float(np.mean(psi_values)) if psi_values else 0.0

        n_features     = len(results["features"])
        n_drifted      = len(drifted_features)
        drift_ratio    = n_drifted / n_features if n_features > 0 else 0.0

        results["summary"] = {
            "global_score":    round(global_psi, 6),
            "global_psi":      round(global_psi, 6),
            "n_features":      n_features,
            "n_drifted":       n_drifted,
            "drift_ratio":     round(drift_ratio, 4),
            "drifted_features": drifted_features,
            "severity":        self._classify_severity(global_psi),
            "retrain_needed":  global_psi > PSI_MODERATE,
        }

        self._results = results
        log.info(
            "Drift analysé — PSI global : %.4f | %d/%d features driftées | sévérité : %s",
            global_psi, n_drifted, n_features, results["summary"]["severity"],
        )

        return DriftReport(results)

    def _analyze_feature(
        self, feature: str, ref_vals: np.ndarray, cur_vals: np.ndarray
    ) -> dict:
        """Analyse le drift d'une feature individuelle."""
        result: dict = {"feature": feature}

        # PSI (toutes features)
        psi = compute_psi(ref_vals, cur_vals)
        result["psi"] = round(psi, 6)

        is_binary = (
            feature in self.BINARY_FEATURES
            or set(np.unique(ref_vals)).issubset({0.0, 1.0})
        )
        is_continuous = (
            feature in self.CONTINUOUS_FEATURES
            and not is_binary
        )

        if is_binary or feature in self.GENRE_FEATURES:
            # Test Chi2 pour les variables binaires / catégorielles
            chi2_stat, p_value, drifted = compute_chi2_test(
                ref_vals.astype(int), cur_vals.astype(int)
            )
            result.update({
                "test":       "chi2",
                "statistic":  round(chi2_stat, 6),
                "p_value":    round(p_value, 6),
                "drifted":    drifted or psi > PSI_MODERATE,
            })
        else:
            # Test KS pour les variables continues
            ks_stat, p_value, drifted = compute_ks_test(ref_vals, cur_vals)
            wass = compute_wasserstein(ref_vals, cur_vals)
            result.update({
                "test":              "ks",
                "statistic":         round(ks_stat, 6),
                "p_value":           round(p_value, 6),
                "wasserstein":       round(wass, 6),
                "drifted":           drifted or psi > PSI_MODERATE,
            })

        # Statistiques descriptives
        result["ref_mean"] = round(float(ref_vals.mean()), 4)
        result["cur_mean"] = round(float(cur_vals.mean()), 4)
        result["ref_std"]  = round(float(ref_vals.std()),  4)
        result["cur_std"]  = round(float(cur_vals.std()),  4)
        result["mean_shift"] = round(float(abs(cur_vals.mean() - ref_vals.mean())), 4)

        return result

    @staticmethod
    def _classify_severity(psi: float) -> str:
        if psi < PSI_NO_DRIFT:
            return "green"
        elif psi < PSI_MODERATE:
            return "orange"
        else:
            return "red"


# ── Rapport de drift ─────────────────────────────────────────────────────────

class DriftReport:
    """
    Rapport de drift — compatible avec l'API Evidently.

    Méthodes :
      .save_json(path)   → sauvegarde en JSON (utilisé par le DAG Airflow)
      .save_html(path)   → rapport HTML autonome avec tableaux et graphes
      .to_dict()         → retourne le dictionnaire complet
      .summary()         → retourne le résumé (global_score, severity…)
    """

    def __init__(self, results: dict):
        self._results = results

    def to_dict(self) -> dict:
        return self._results

    def summary(self) -> dict:
        return self._results.get("summary", {})

    @property
    def global_score(self) -> float:
        return self._results.get("summary", {}).get("global_score", 0.0)

    @property
    def severity(self) -> str:
        return self._results.get("summary", {}).get("severity", "green")

    @property
    def retrain_needed(self) -> bool:
        return self._results.get("summary", {}).get("retrain_needed", False)

    def save_json(self, path: str) -> str:
        """Sauvegarde le rapport en JSON (utilisé par trigger_kubeflow_dag)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._results, f, indent=2, default=str)
        log.info("Rapport JSON sauvegardé : %s", path)
        return path

    def save_html(self, path: str) -> str:
        """Génère un rapport HTML standalone avec tableaux et mini-graphes ASCII."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html()
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        log.info("Rapport HTML sauvegardé : %s", path)
        return path

    def _build_html(self) -> str:
        summary  = self._results.get("summary", {})
        features = self._results.get("features", {})
        ts       = self._results.get("timestamp", "")
        dataset  = self._results.get("dataset", "")

        severity_color = {"green": "#27ae60", "orange": "#e67e22", "red": "#e74c3c"}
        sev   = summary.get("severity", "green")
        color = severity_color.get(sev, "#27ae60")

        rows = ""
        for feat, info in sorted(features.items(),
                                  key=lambda x: x[1].get("psi", 0), reverse=True):
            psi     = info.get("psi",     0)
            drifted = info.get("drifted", False)
            test    = info.get("test",    "-")
            p_val   = info.get("p_value", 1.0)
            stat    = info.get("statistic", 0.0)
            shift   = info.get("mean_shift", 0.0)

            psi_color = severity_color[DriftDetector._classify_severity(psi)]
            drift_badge = (
                '<span style="color:#e74c3c;font-weight:bold">⚠ DRIFT</span>'
                if drifted else
                '<span style="color:#27ae60">✓ OK</span>'
            )

            rows += f"""
            <tr>
              <td><strong>{feat}</strong></td>
              <td style="color:{psi_color};font-weight:bold">{psi:.4f}</td>
              <td>{test.upper()}</td>
              <td>{stat:.4f}</td>
              <td>{p_val:.4f}</td>
              <td>{shift:.4f}</td>
              <td>{drift_badge}</td>
            </tr>"""

        return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>Rapport de Drift — {dataset}</title>
  <style>
    body {{ font-family: 'Segoe UI', sans-serif; margin: 32px; background: #f8f9fa; }}
    h1   {{ color: #2c3e50; border-bottom: 3px solid {color}; padding-bottom: 8px; }}
    h2   {{ color: #34495e; margin-top: 32px; }}
    .summary-card {{
      background: white; border-radius: 8px; padding: 20px 28px;
      box-shadow: 0 2px 8px rgba(0,0,0,.1); margin-bottom: 24px;
      border-left: 5px solid {color};
    }}
    .metric {{ display: inline-block; margin: 8px 24px 8px 0; }}
    .metric .value {{ font-size: 2em; font-weight: bold; color: {color}; }}
    .metric .label {{ font-size: 0.85em; color: #7f8c8d; }}
    table {{ border-collapse: collapse; width: 100%; background: white;
             border-radius: 8px; overflow: hidden;
             box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
    th    {{ background: #2c3e50; color: white; padding: 12px 16px; text-align: left; }}
    td    {{ padding: 10px 16px; border-bottom: 1px solid #ecf0f1; }}
    tr:hover {{ background: #f0f3f7; }}
    .badge-green  {{ background:#eafaf1; color:#27ae60; padding:3px 10px; border-radius:12px; }}
    .badge-orange {{ background:#fef9e7; color:#e67e22; padding:3px 10px; border-radius:12px; }}
    .badge-red    {{ background:#fdedec; color:#e74c3c; padding:3px 10px; border-radius:12px; font-weight:bold; }}
    footer {{ margin-top: 40px; color: #bdc3c7; font-size: 0.8em; text-align: center; }}
  </style>
</head>
<body>
  <h1>📊 Rapport de Drift des Données</h1>
  <p style="color:#7f8c8d">Dataset : <strong>{dataset}</strong> &nbsp;|&nbsp; Généré : {ts[:19]}</p>

  <div class="summary-card">
    <div class="metric">
      <div class="value">{summary.get('global_psi', 0):.4f}</div>
      <div class="label">PSI global</div>
    </div>
    <div class="metric">
      <div class="value">{summary.get('n_drifted', 0)}/{summary.get('n_features', 0)}</div>
      <div class="label">Features driftées</div>
    </div>
    <div class="metric">
      <div class="value" style="color:{color}">{sev.upper()}</div>
      <div class="label">Sévérité</div>
    </div>
    <div class="metric">
      <div class="value">{self._results.get('n_reference', 0):,}</div>
      <div class="label">Données référence</div>
    </div>
    <div class="metric">
      <div class="value">{self._results.get('n_current', 0):,}</div>
      <div class="label">Données courantes</div>
    </div>
    {"<p style='margin-top:16px;font-size:1.1em'><strong>⚠️ Ré-entraînement recommandé</strong> — PSI &gt; 0.25</p>" if self.retrain_needed else "<p style='margin-top:16px;font-size:1.1em'>✅ Distribution stable — pas de ré-entraînement nécessaire</p>"}
  </div>

  <h2>Analyse par feature</h2>
  <table>
    <thead>
      <tr>
        <th>Feature</th><th>PSI</th><th>Test</th><th>Statistique</th>
        <th>P-value</th><th>Δ Moyenne</th><th>Statut</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>

  <footer>MovieLens MLOps · Monitoring de drift · Période 7</footer>
</body>
</html>"""
