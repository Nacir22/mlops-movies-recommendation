"""
feast/features/user_features.py
Feature Views Feast pour les utilisateurs et les films MovieLens.

Deux Feature Views :
  - user_feature_view  : profil utilisateur + stats comportementales
  - movie_feature_view : métadonnées film + stats de popularité

Architecture offline / online :
  ┌──────────────────────────────────────────────────────────┐
  │  Offline Store (fichiers CSV/Parquet sur MinIO)          │
  │    → Entraînement : get_historical_features()            │
  │                                                          │
  │  Online Store (Redis)                                    │
  │    → Inférence temps réel : get_online_features()        │
  │    → Alimenté par : feast materialize-incremental        │
  └──────────────────────────────────────────────────────────┘
"""
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "feast" / "offline"

try:
    from datetime import timedelta
    from feast import FeatureView, Field, FileSource
    from feast.types import Float32, Float64, Int64

    # ── Sources de données ──────────────────────────────────────────
    user_source = FileSource(
        path=str(DATA_DIR / "user_features.csv"),
        timestamp_field="event_timestamp",
        created_timestamp_column="created",
    )

    movie_source = FileSource(
        path=str(DATA_DIR / "movie_features.csv"),
        timestamp_field="event_timestamp",
        created_timestamp_column="created",
    )

    # ── Feature View Utilisateurs ───────────────────────────────────
    user_feature_view = FeatureView(
        name="user_features",
        entities=["user_id"],
        ttl=timedelta(days=7),
        schema=[
            Field(name="gender_enc",       dtype=Int64,   description="Genre encodé (0=F, 1=M)"),
            Field(name="age",              dtype=Int64,   description="Tranche d'âge"),
            Field(name="occupation",       dtype=Int64,   description="Code de profession (0-20)"),
            Field(name="user_avg_rating",  dtype=Float64, description="Note moyenne de l'utilisateur"),
            Field(name="user_n_ratings",   dtype=Int64,   description="Nombre de films notés"),
            Field(name="user_std_rating",  dtype=Float64, description="Écart-type des notes"),
        ],
        online=True,
        source=user_source,
        tags={"team": "mlops", "domain": "users", "version": "1.0"},
    )

    # ── Feature View Films ──────────────────────────────────────────
    movie_feature_view = FeatureView(
        name="movie_features",
        entities=["movie_id"],
        ttl=timedelta(days=30),
        schema=[
            Field(name="year",             dtype=Float32, description="Année de sortie"),
            Field(name="movie_avg_rating", dtype=Float64, description="Note moyenne du film"),
            Field(name="movie_n_ratings",  dtype=Int64,   description="Nombre de notes reçues"),
            Field(name="action",           dtype=Float32, description="Genre Action"),
            Field(name="adventure",        dtype=Float32, description="Genre Adventure"),
            Field(name="animation",        dtype=Float32, description="Genre Animation"),
            Field(name="childrens",        dtype=Float32, description="Genre Children's"),
            Field(name="comedy",           dtype=Float32, description="Genre Comedy"),
            Field(name="crime",            dtype=Float32, description="Genre Crime"),
            Field(name="documentary",      dtype=Float32, description="Genre Documentary"),
            Field(name="drama",            dtype=Float32, description="Genre Drama"),
            Field(name="fantasy",          dtype=Float32, description="Genre Fantasy"),
            Field(name="film_noir",        dtype=Float32, description="Genre Film-Noir"),
            Field(name="horror",           dtype=Float32, description="Genre Horror"),
            Field(name="musical",          dtype=Float32, description="Genre Musical"),
            Field(name="mystery",          dtype=Float32, description="Genre Mystery"),
            Field(name="romance",          dtype=Float32, description="Genre Romance"),
            Field(name="sci_fi",           dtype=Float32, description="Genre Sci-Fi"),
            Field(name="thriller",         dtype=Float32, description="Genre Thriller"),
            Field(name="war",              dtype=Float32, description="Genre War"),
            Field(name="western",          dtype=Float32, description="Genre Western"),
        ],
        online=True,
        source=movie_source,
        tags={"team": "mlops", "domain": "movies", "version": "1.0"},
    )

    print("✅ Feature Views Feast définis : user_features, movie_features")

except ImportError:
    print("ℹ️  Feast non installé — définitions documentaires uniquement")
    print("   Installer avec : pip install feast")
    print("   Feature Views : user_features (6 champs), movie_features (21 champs)")
    user_feature_view  = None
    movie_feature_view = None
