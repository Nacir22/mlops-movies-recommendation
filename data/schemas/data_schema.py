"""
data/schemas/data_schema.py
Schémas de validation des données MovieLens 1M.
Utilisé pour valider les fichiers .dat avant traitement.
"""
from dataclasses import dataclass
from typing import Literal

# ── Constantes MovieLens ────────────────────────────────────────────────────
VALID_RATINGS     = {1, 2, 3, 4, 5}
VALID_GENDERS     = {"M", "F"}
VALID_AGES        = {1, 18, 25, 35, 45, 50, 56}
VALID_OCCUPATIONS = set(range(21))
LIKE_THRESHOLD    = 4   # Note ≥ 4 → film "aimé"

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
]


@dataclass
class MovieSchema:
    """Format : MovieID::Title::Genres"""
    movie_id: int
    title:    str
    genres:   str   # séparés par "|"

    def validate(self) -> bool:
        assert self.movie_id > 0,    "MovieID doit être positif"
        assert len(self.title) > 0,  "Title vide"
        for g in self.genres.split("|"):
            assert g in ALL_GENRES,  f"Genre inconnu : {g}"
        return True


@dataclass
class UserSchema:
    """Format : UserID::Gender::Age::Occupation::Zip-code"""
    user_id:    int
    gender:     Literal["M", "F"]
    age:        int
    occupation: int
    zip_code:   str

    def validate(self) -> bool:
        assert self.user_id > 0,                      "UserID doit être positif"
        assert self.gender in VALID_GENDERS,           f"Gender invalide : {self.gender}"
        assert self.age in VALID_AGES,                 f"Age invalide : {self.age}"
        assert self.occupation in VALID_OCCUPATIONS,   f"Occupation invalide : {self.occupation}"
        return True


@dataclass
class RatingSchema:
    """Format : UserID::MovieID::Rating::Timestamp"""
    user_id:   int
    movie_id:  int
    rating:    int
    timestamp: int

    def validate(self) -> bool:
        assert self.user_id > 0,              "UserID doit être positif"
        assert self.movie_id > 0,             "MovieID doit être positif"
        assert self.rating in VALID_RATINGS,  f"Rating invalide : {self.rating}"
        assert self.timestamp > 0,            "Timestamp doit être positif"
        return True


def validate_dat_file(filepath: str, schema_class, sep: str = "::") -> dict:
    """
    Valide un fichier .dat MovieLens ligne par ligne.
    Retourne un dict avec le nombre de lignes valides/invalides.
    """
    valid, invalid, errors = 0, 0, []

    with open(filepath, encoding="latin-1") as f:
        for i, line in enumerate(f, 1):
            parts = line.strip().split(sep)
            try:
                if schema_class == MovieSchema:
                    obj = MovieSchema(int(parts[0]), parts[1], parts[2])
                elif schema_class == UserSchema:
                    obj = UserSchema(int(parts[0]), parts[1], int(parts[2]),
                                     int(parts[3]), parts[4])
                elif schema_class == RatingSchema:
                    obj = RatingSchema(int(parts[0]), int(parts[1]),
                                       int(parts[2]), int(parts[3]))
                obj.validate()
                valid += 1
            except Exception as e:
                invalid += 1
                if len(errors) < 5:   # Garder les 5 premières erreurs
                    errors.append(f"Ligne {i}: {e}")

    return {"valid": valid, "invalid": invalid, "errors": errors}


if __name__ == "__main__":
    import os
    raw_dir = os.path.join(os.path.dirname(__file__), "..", "raw")

    for filename, schema in [
        ("movies.dat",  MovieSchema),
        ("users.dat",   UserSchema),
        ("ratings.dat", RatingSchema),
    ]:
        path = os.path.join(raw_dir, filename)
        if os.path.exists(path):
            result = validate_dat_file(path, schema)
            status = "✅" if result["invalid"] == 0 else "⚠️ "
            print(f"{status} {filename}: {result['valid']} valides, {result['invalid']} invalides")
            for e in result["errors"]:
                print(f"   {e}")
        else:
            print(f"❌ {filename} introuvable : {path}")
