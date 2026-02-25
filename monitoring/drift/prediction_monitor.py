"""
monitoring/drift/prediction_monitor.py
Monitoring de la distribution des prédictions en production.

Surveille :
  - Distribution des scores P(Liked) : drift vs référence
  - Taux de prédictions positives (will_like=True) dans le temps
  - Latence de l'API : P50, P95, P99
  - Alertes automatiques si anomalie détectée

Les métriques sont agrégées par fenêtre temporelle (heure, jour)
et sauvegardées en JSON pour être lues par les DAGs Airflow.
"""
from __future__ import annotations

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from monitoring.drift.drift_detector import (
    compute_psi, compute_ks_test, PSI_NO_DRIFT, PSI_MODERATE
)

log = logging.getLogger(__name__)

METRICS_DIR = Path(__file__).parent.parent / "alerts"


class PredictionMonitor:
    """
    Surveille la distribution des prédictions de l'API en production.

    Utilisation :
        monitor = PredictionMonitor.from_reference(reference_scores)
        monitor.add_batch(current_scores)
        report  = monitor.analyze()
        report.save("monitoring/alerts/pred_monitor_20240115.json")
    """

    def __init__(
        self,
        reference_scores: np.ndarray,
        window_hours:     int = 24,
    ):
        """
        Parameters
        ----------
        reference_scores : scores P(Liked) sur le jeu de validation référence
        window_hours     : fenêtre d'agrégation pour les métriques temporelles
        """
        self.reference_scores = np.asarray(reference_scores)
        self.window_hours     = window_hours
        self._batches: list[dict] = []

        # Statistiques de référence
        self.ref_stats = self._compute_score_stats(self.reference_scores)
        log.info(
            "PredictionMonitor initialisé — ref μ=%.4f σ=%.4f pos_rate=%.3f",
            self.ref_stats["mean"], self.ref_stats["std"],
            self.ref_stats["positive_rate"],
        )

    @classmethod
    def from_reference(
        cls, reference_data: pd.DataFrame, model, feature_columns: list[str]
    ) -> "PredictionMonitor":
        """Crée un monitor à partir des prédictions sur le jeu de référence."""
        proba = model.predict_proba(reference_data[feature_columns])[:, 1]
        return cls(proba)

    def add_batch(
        self,
        scores:     np.ndarray,
        latencies_ms: Optional[np.ndarray] = None,
        timestamp:  Optional[datetime] = None,
    ) -> "PredictionMonitor":
        """
        Ajoute un batch de prédictions de production.

        Parameters
        ----------
        scores       : probabilités P(Liked) pour ce batch
        latencies_ms : latences en ms (optionnel)
        timestamp    : horodatage du batch (défaut : maintenant)
        """
        ts     = timestamp or datetime.utcnow()
        scores = np.asarray(scores)

        batch = {
            "timestamp":    ts.isoformat(),
            "n":            len(scores),
            "stats":        self._compute_score_stats(scores),
            "psi":          compute_psi(self.reference_scores, scores),
        }

        if latencies_ms is not None:
            lats = np.asarray(latencies_ms)
            batch["latency"] = {
                "p50": float(np.percentile(lats, 50)),
                "p95": float(np.percentile(lats, 95)),
                "p99": float(np.percentile(lats, 99)),
                "mean": float(lats.mean()),
            }

        self._batches.append(batch)
        return self

    def analyze(self) -> "PredictionMonitorReport":
        """Analyse tous les batches accumulés et retourne un rapport."""
        if not self._batches:
            raise ValueError("Aucun batch ajouté — appelez add_batch() d'abord")

        # Agréger tous les scores de production
        all_psis  = [b["psi"] for b in self._batches]
        all_stats = [b["stats"] for b in self._batches]

        mean_psi       = float(np.mean(all_psis))
        max_psi        = float(np.max(all_psis))
        pos_rates      = [s["positive_rate"] for s in all_stats]
        mean_pos_rate  = float(np.mean(pos_rates))

        # Dérive du taux positif vs référence
        pos_rate_drift = abs(mean_pos_rate - self.ref_stats["positive_rate"])

        # Alertes
        alerts = []
        if mean_psi > PSI_MODERATE:
            alerts.append({
                "type":     "score_distribution_drift",
                "severity": "red",
                "message":  f"PSI moyen des scores : {mean_psi:.4f} > {PSI_MODERATE}",
                "value":    mean_psi,
            })
        elif mean_psi > PSI_NO_DRIFT:
            alerts.append({
                "type":     "score_distribution_drift",
                "severity": "orange",
                "message":  f"PSI moyen des scores : {mean_psi:.4f} > {PSI_NO_DRIFT}",
                "value":    mean_psi,
            })

        if pos_rate_drift > 0.15:
            alerts.append({
                "type":     "positive_rate_shift",
                "severity": "orange" if pos_rate_drift < 0.25 else "red",
                "message":  (
                    f"Taux positif courant : {mean_pos_rate:.3f} "
                    f"vs référence : {self.ref_stats['positive_rate']:.3f} "
                    f"(δ={pos_rate_drift:.3f})"
                ),
                "value":    pos_rate_drift,
            })

        # Latences (si disponibles)
        latency_batches = [b["latency"] for b in self._batches if "latency" in b]
        latency_summary = None
        if latency_batches:
            p99_values = [b["p99"] for b in latency_batches]
            latency_summary = {
                "p50":  float(np.mean([b["p50"]  for b in latency_batches])),
                "p95":  float(np.mean([b["p95"]  for b in latency_batches])),
                "p99":  float(np.mean(p99_values)),
                "max_p99": float(max(p99_values)),
            }
            if latency_summary["p99"] > 500:
                alerts.append({
                    "type":     "high_latency",
                    "severity": "warning",
                    "message":  f"P99 latence : {latency_summary['p99']:.0f}ms > 500ms",
                    "value":    latency_summary["p99"],
                })

        report_data = {
            "timestamp":        datetime.utcnow().isoformat(),
            "n_batches":        len(self._batches),
            "total_predictions": sum(b["n"] for b in self._batches),
            "reference_stats":  self.ref_stats,
            "current_summary": {
                "mean_psi":       round(mean_psi, 6),
                "max_psi":        round(max_psi, 6),
                "mean_pos_rate":  round(mean_pos_rate, 4),
                "pos_rate_drift": round(pos_rate_drift, 4),
            },
            "latency":   latency_summary,
            "alerts":    alerts,
            "batches":   self._batches[-10:],   # 10 derniers batches seulement
            "retrain_needed": any(a["severity"] == "red" for a in alerts),
        }

        return PredictionMonitorReport(report_data)

    @staticmethod
    def _compute_score_stats(scores: np.ndarray) -> dict:
        return {
            "n":             len(scores),
            "mean":          round(float(scores.mean()),   4),
            "std":           round(float(scores.std()),    4),
            "min":           round(float(scores.min()),    4),
            "max":           round(float(scores.max()),    4),
            "p25":           round(float(np.percentile(scores, 25)), 4),
            "p50":           round(float(np.percentile(scores, 50)), 4),
            "p75":           round(float(np.percentile(scores, 75)), 4),
            "positive_rate": round(float((scores >= 0.5).mean()), 4),
        }


class PredictionMonitorReport:
    """Rapport du monitoring de prédictions."""

    def __init__(self, data: dict):
        self._data = data

    def to_dict(self) -> dict:
        return self._data

    @property
    def retrain_needed(self) -> bool:
        return self._data.get("retrain_needed", False)

    @property
    def alerts(self) -> list[dict]:
        return self._data.get("alerts", [])

    def save(self, path: str) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)
        log.info("Rapport monitoring sauvegardé : %s", path)
        return path

    def print_summary(self):
        d = self._data
        print(f"\n{'═'*55}")
        print("📡 MONITORING DES PRÉDICTIONS")
        print(f"{'═'*55}")
        print(f"  Batches analysés   : {d['n_batches']}")
        print(f"  Prédictions totales: {d['total_predictions']:,}")
        cur = d["current_summary"]
        ref = d["reference_stats"]
        print(f"\n  PSI moyen scores   : {cur['mean_psi']:.4f}", end="")
        if cur['mean_psi'] > PSI_MODERATE:
            print("  ⚠️  DRIFT FORT")
        elif cur['mean_psi'] > PSI_NO_DRIFT:
            print("  ⚠️  Drift modéré")
        else:
            print("  ✅ stable")
        print(f"  Taux positif réf   : {ref['positive_rate']:.3f}")
        print(f"  Taux positif actuel: {cur['mean_pos_rate']:.3f} (δ={cur['pos_rate_drift']:.3f})")

        if d.get("latency"):
            lat = d["latency"]
            print(f"\n  Latence P50/P95/P99: {lat['p50']:.0f}/{lat['p95']:.0f}/{lat['p99']:.0f} ms")

        if d["alerts"]:
            print(f"\n  🚨 {len(d['alerts'])} alerte(s) :")
            for a in d["alerts"]:
                icon = "🔴" if a["severity"]=="red" else "🟠"
                print(f"    {icon} [{a['type']}] {a['message']}")
        else:
            print("\n  ✅ Aucune alerte")

        print(f"\n  Ré-entraînement    : {'⚠️  OUI' if self.retrain_needed else '✅ NON'}")
        print(f"{'═'*55}")
