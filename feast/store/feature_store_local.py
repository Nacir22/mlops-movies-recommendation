"""
feast/store/feature_store_local.py
Simulateur local du Feature Store Feast.

Reproduit exactement l'API de Feast (get_historical_features,
get_online_features, materialize) sans dépendances réseau.
En production, remplacer les appels à LocalFeatureStore
par les appels Feast natifs — les signatures sont identiques.

Architecture :
  ┌─────────────────────────────────────────────────────────────┐
  │  OFFLINE STORE  (CSV locaux → production : Parquet/MinIO)   │
  │    get_historical_features(entity_df, features)             │
  │    → retourne un DataFrame pour l'entraînement              │
  │                                                             │
  │  ONLINE STORE   (dict en mémoire → production : Redis)      │
  │    materialize()                     offline → online       │
  │    get_online_features(entities, features)                  │
  │    → retourne un dict pour l'inférence temps réel           │
  └─────────────────────────────────────────────────────────────┘
"""
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent.parent / "data" / "feast"
OFFLINE    = DATA_DIR / "offline"
ONLINE_DIR = DATA_DIR / "online"

# Mapping feature_view → fichier CSV offline
FEATURE_VIEW_FILES = {
    "user_features":  OFFLINE / "user_features.csv",
    "movie_features": OFFLINE / "movie_features.csv",
}

# Clé primaire par feature view
ENTITY_KEYS = {
    "user_features":  "user_id",
    "movie_features": "movie_id",
}

# Fichiers online (snapshot matérialisé)
ONLINE_FILES = {
    "user_features":  ONLINE_DIR / "user_features_online.csv",
    "movie_features": ONLINE_DIR / "movie_features_online.csv",
}


class LocalFeatureStore:
    """
    Simulateur du Feature Store Feast.
    API identique à feast.FeatureStore pour faciliter la migration.
    """

    def __init__(self, repo_path: str = None):
        self.repo_path = Path(repo_path) if repo_path else Path(__file__).parent.parent
        self._online_cache: dict = {}   # user_features:{id} → {feat: val}
        self._materialized  = False
        logger.info("LocalFeatureStore initialisé (mode local sans Redis)")

    # ── OFFLINE STORE ──────────────────────────────────────────────────────

    def get_historical_features(
        self,
        entity_df:   pd.DataFrame,
        features:    list[str],
        full_feature_names: bool = False,
    ) -> pd.DataFrame:
        """
        Récupère les features historiques pour un ensemble d'entités.
        Équivalent Feast : store.get_historical_features(entity_df, features).to_df()

        Paramètres
        ----------
        entity_df : DataFrame avec les colonnes d'entités + 'event_timestamp'
            Ex : UserID, MovieID, event_timestamp
        features  : liste au format "feature_view:feature_name"
            Ex : ["user_features:gender_enc", "movie_features:year"]

        Retourne
        --------
        DataFrame entity_df enrichi des features demandées
        """
        result = entity_df.copy()

        # Grouper les features par feature view
        fv_requests: dict[str, list[str]] = {}
        for feat_ref in features:
            fv_name, feat_name = feat_ref.split(":")
            fv_requests.setdefault(fv_name, []).append(feat_name)

        for fv_name, feat_names in fv_requests.items():
            fv_file = FEATURE_VIEW_FILES.get(fv_name)
            if fv_file is None or not fv_file.exists():
                logger.warning(f"Feature view '{fv_name}' introuvable : {fv_file}")
                continue

            fv_df      = pd.read_csv(fv_file)
            entity_key = ENTITY_KEYS[fv_name]
            entity_col = entity_key.upper() if entity_key.upper() in result.columns \
                         else entity_key

            # Colonnes à récupérer
            cols = [entity_key] + [f for f in feat_names if f in fv_df.columns]
            fv_subset = fv_df[cols].rename(columns={entity_key: entity_col})

            result = result.merge(fv_subset, on=entity_col, how="left")

        result = result.fillna(0)
        logger.info(f"get_historical_features → {result.shape} ({len(features)} features)")
        return result

    # ── MATÉRIALISATION : offline → online ────────────────────────────────

    def materialize(
        self,
        start_date: Optional[datetime] = None,
        end_date:   Optional[datetime] = None,
    ) -> dict:
        """
        Matérialise les features de l'offline store vers l'online store.
        Équivalent Feast : store.materialize(start_date, end_date)

        En production, cette commande copie les features dans Redis.
        Ici, on charge les CSV en mémoire et on les sauvegarde en JSON.
        """
        end_date   = end_date   or datetime.utcnow()
        start_date = start_date or (end_date - timedelta(days=7))

        stats = {}
        self._online_cache = {}

        for fv_name, fv_file in FEATURE_VIEW_FILES.items():
            if not fv_file.exists():
                logger.warning(f"Fichier offline manquant : {fv_file}")
                continue

            df         = pd.read_csv(fv_file)
            entity_key = ENTITY_KEYS[fv_name]

            # Filtrer par timestamp si disponible
            if "event_timestamp" in df.columns:
                df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
                mask = (df["event_timestamp"] >= pd.Timestamp(start_date)) & \
                       (df["event_timestamp"] <= pd.Timestamp(end_date))
                df = df[mask] if mask.any() else df   # fallback : tout prendre

            # Charger dans le cache mémoire
            feat_cols = [c for c in df.columns
                         if c not in [entity_key, "event_timestamp", "created"]]
            self._online_cache[fv_name] = {}

            for _, row in df.iterrows():
                key = int(row[entity_key])
                self._online_cache[fv_name][key] = {
                    c: float(row[c]) for c in feat_cols
                }

            # Sauvegarder le snapshot online (JSON)
            snapshot_path = ONLINE_DIR / f"{fv_name}_snapshot.json"
            with open(snapshot_path, "w") as f:
                # Convertir les clés int en str pour JSON
                json.dump(
                    {str(k): v for k, v in self._online_cache[fv_name].items()},
                    f, indent=2
                )

            stats[fv_name] = len(self._online_cache[fv_name])
            logger.info(f"Matérialisé {fv_name} : {stats[fv_name]} entités")

        self._materialized = True
        print("✅ Matérialisation terminée :")
        for fv, n in stats.items():
            print(f"   {fv:20s} : {n:,} entités dans l'online store")
        return stats

    def materialize_incremental(self, end_date: datetime = None) -> dict:
        """
        Matérialisation incrémentale (nouvelles données seulement).
        Équivalent Feast : store.materialize_incremental(end_date)
        """
        end_date   = end_date or datetime.utcnow()
        start_date = end_date - timedelta(days=1)   # Delta d'1 jour par défaut
        print(f"🔄 Matérialisation incrémentale : {start_date.date()} → {end_date.date()}")
        return self.materialize(start_date, end_date)

    # ── ONLINE STORE ───────────────────────────────────────────────────────

    def get_online_features(
        self,
        features:     list[str],
        entity_rows:  list[dict],
    ) -> "OnlineResponse":
        """
        Récupère les features depuis l'online store pour l'inférence.
        Équivalent Feast : store.get_online_features(features, entity_rows).to_dict()

        Paramètres
        ----------
        features     : ["user_features:gender_enc", "movie_features:year"]
        entity_rows  : [{"user_id": 1, "movie_id": 10}, ...]

        Retourne
        --------
        OnlineResponse avec méthode .to_dict() et .to_df()
        """
        # Charger le cache si pas encore matérialisé
        if not self._materialized:
            self._load_online_cache()

        # Grouper les features par feature view
        fv_requests: dict[str, list[str]] = {}
        for feat_ref in features:
            fv_name, feat_name = feat_ref.split(":")
            fv_requests.setdefault(fv_name, []).append(feat_name)

        result_rows = []
        for entity_row in entity_rows:
            row = dict(entity_row)   # copier les clés d'entités

            for fv_name, feat_names in fv_requests.items():
                entity_key = ENTITY_KEYS[fv_name]
                entity_val = entity_row.get(entity_key)

                fv_cache  = self._online_cache.get(fv_name, {})
                fv_record = fv_cache.get(int(entity_val), {}) if entity_val else {}

                for feat in feat_names:
                    col = feat if not False else f"{fv_name}__{feat}"
                    row[col] = fv_record.get(feat, 0.0)

            result_rows.append(row)

        return OnlineResponse(result_rows)

    def _load_online_cache(self):
        """Charge les snapshots JSON sauvegardés par materialize()."""
        for fv_name in FEATURE_VIEW_FILES:
            snapshot_path = ONLINE_DIR / f"{fv_name}_snapshot.json"
            if snapshot_path.exists():
                with open(snapshot_path) as f:
                    raw = json.load(f)
                self._online_cache[fv_name] = {int(k): v for k, v in raw.items()}
                logger.info(f"Cache online chargé : {fv_name} ({len(raw)} entités)")
        self._materialized = True

    # ── Méthodes utilitaires ───────────────────────────────────────────────

    def list_feature_views(self) -> list:
        return list(FEATURE_VIEW_FILES.keys())

    def get_feature_view_stats(self) -> dict:
        stats = {}
        for fv_name, fv_file in FEATURE_VIEW_FILES.items():
            if fv_file.exists():
                df = pd.read_csv(fv_file)
                stats[fv_name] = {"rows": len(df), "columns": list(df.columns)}
        return stats


class OnlineResponse:
    """Réponse de get_online_features(), compatible avec l'API Feast."""

    def __init__(self, rows: list[dict]):
        self._rows = rows

    def to_dict(self) -> dict:
        """Retourne {feature_name: [val1, val2, ...]}"""
        if not self._rows:
            return {}
        keys = self._rows[0].keys()
        return {k: [r[k] for r in self._rows] for k in keys}

    def to_df(self) -> pd.DataFrame:
        """Retourne un DataFrame une ligne par entité."""
        return pd.DataFrame(self._rows)
