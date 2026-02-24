"""
feast/entities/entities.py
Entités Feast pour le projet MovieLens Recommender.

Une entité est la clé primaire qui relie les feature views entre elles
et avec la table d'entraînement.
"""

try:
    from feast import Entity, ValueType

    user_entity = Entity(
        name="user_id",
        value_type=ValueType.INT64,
        description="Identifiant unique de l'utilisateur MovieLens (1..6040)",
        tags={"owner": "mlops-team", "domain": "users"},
    )

    movie_entity = Entity(
        name="movie_id",
        value_type=ValueType.INT64,
        description="Identifiant unique du film MovieLens (1..3952)",
        tags={"owner": "mlops-team", "domain": "movies"},
    )

except ImportError:
    # Feast non installé — définitions documentaires
    print("ℹ️  Feast non installé. Installer avec : pip install feast")
    print("   Entités définies : user_id (INT64), movie_id (INT64)")

    class _MockEntity:
        def __init__(self, name, **kwargs):
            self.name = name
            self.__dict__.update(kwargs)
        def __repr__(self):
            return f"Entity(name={self.name})"

    user_entity  = _MockEntity("user_id",  value_type="INT64")
    movie_entity = _MockEntity("movie_id", value_type="INT64")
