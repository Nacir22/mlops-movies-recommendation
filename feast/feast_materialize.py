"""
feast/feast_materialize.py
Script de matérialisation des features : offline store → online store.

Usage :
    # Matérialisation complète
    python feast/feast_materialize.py

    # Matérialisation incrémentale (nouvelles données seulement)
    python feast/feast_materialize.py --incremental

    # Avec Feast natif (si installé)
    cd feast/ && feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")

Ce script peut aussi être appelé directement depuis un DAG Airflow.
"""
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_with_feast_native(incremental: bool = True):
    """Matérialisation via Feast natif (si installé)."""
    import feast
    from feast import FeatureStore

    store = FeatureStore(repo_path=str(Path(__file__).parent))
    end_date = datetime.utcnow()

    if incremental:
        print(f"🔄 feast materialize-incremental → {end_date.strftime('%Y-%m-%dT%H:%M:%S')}")
        store.materialize_incremental(end_date=end_date)
    else:
        from datetime import timedelta
        start_date = end_date - timedelta(days=30)
        print(f"📦 feast materialize {start_date.date()} → {end_date.date()}")
        store.materialize(start_date=start_date, end_date=end_date)

    print("✅ Matérialisation Feast terminée")


def run_with_local_store(incremental: bool = True):
    """Matérialisation via le simulateur local."""
    from feast.store.feature_store_local import LocalFeatureStore

    store    = LocalFeatureStore()
    end_date = datetime.utcnow()

    if incremental:
        stats = store.materialize_incremental(end_date=end_date)
    else:
        stats = store.materialize(end_date=end_date)

    return stats


def validate_online_store():
    """Vérifie que l'online store contient des données valides."""
    from feast.store.feature_store_local import LocalFeatureStore

    store = LocalFeatureStore()

    # Test : récupérer les features d'un utilisateur connu
    response = store.get_online_features(
        features=[
            "user_features:gender_enc",
            "user_features:age",
            "user_features:user_avg_rating",
            "movie_features:year",
            "movie_features:movie_avg_rating",
        ],
        entity_rows=[
            {"user_id": 1,  "movie_id": 1},
            {"user_id": 42, "movie_id": 10},
        ]
    )

    df = response.to_df()
    print("\n🔍 Validation online store :")
    print(df.to_string())

    # Vérifications
    assert len(df) == 2,                     "Doit retourner 2 lignes"
    assert "gender_enc" in df.columns,       "gender_enc manquant"
    assert "user_avg_rating" in df.columns,  "user_avg_rating manquant"
    assert "movie_avg_rating" in df.columns, "movie_avg_rating manquant"
    assert df["user_avg_rating"].notna().all(), "Valeurs nulles dans user_avg_rating"

    print("\n✅ Online store validé — toutes les features sont disponibles")
    return True


def main():
    parser = argparse.ArgumentParser(description="Matérialisation Feast — MovieLens")
    parser.add_argument("--incremental", action="store_true", default=True,
                        help="Matérialisation incrémentale (défaut)")
    parser.add_argument("--full",        action="store_true",
                        help="Matérialisation complète (30 derniers jours)")
    parser.add_argument("--validate",    action="store_true",
                        help="Valider l'online store après matérialisation")
    args = parser.parse_args()

    incremental = not args.full

    print("=" * 60)
    print("🌾 Matérialisation Feature Store — MovieLens Recommender")
    print("=" * 60)
    print(f"Mode : {'incrémentale' if incremental else 'complète'}")
    print(f"Date : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print()

    # Essayer Feast natif, fallback sur simulateur local
    try:
        import feast
        print(f"🎯 Feast {feast.__version__} détecté — utilisation native")
        run_with_feast_native(incremental)
    except ImportError:
        print("⚙️  Feast non installé — utilisation du simulateur local")
        run_with_local_store(incremental)

    if args.validate:
        validate_online_store()

    print("\n🎉 Matérialisation terminée avec succès")


if __name__ == "__main__":
    main()
