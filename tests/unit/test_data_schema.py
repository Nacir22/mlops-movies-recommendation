"""
tests/unit/test_data_schema.py
Tests unitaires des schémas de validation MovieLens.
"""
import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from data.schemas.data_schema import (
    MovieSchema, UserSchema, RatingSchema,
    validate_dat_file, VALID_RATINGS, VALID_AGES, ALL_GENRES,
)


class TestMovieSchema:
    def test_valid_movie(self):
        m = MovieSchema(1, "Toy Story (1995)", "Animation|Comedy")
        assert m.validate()

    def test_invalid_genre(self):
        with pytest.raises(AssertionError):
            MovieSchema(1, "Test (2000)", "UnknownGenre").validate()

    def test_invalid_movie_id(self):
        with pytest.raises(AssertionError):
            MovieSchema(0, "Test (2000)", "Drama").validate()

    def test_empty_title(self):
        with pytest.raises(AssertionError):
            MovieSchema(1, "", "Drama").validate()


class TestUserSchema:
    def test_valid_user_male(self):
        u = UserSchema(1, "M", 25, 4, "10001")
        assert u.validate()

    def test_valid_user_female(self):
        u = UserSchema(2, "F", 18, 0, "90210")
        assert u.validate()

    def test_invalid_gender(self):
        with pytest.raises(AssertionError):
            UserSchema(1, "X", 25, 4, "10001").validate()

    def test_invalid_age(self):
        with pytest.raises(AssertionError):
            UserSchema(1, "M", 30, 4, "10001").validate()   # 30 pas dans VALID_AGES

    def test_invalid_occupation(self):
        with pytest.raises(AssertionError):
            UserSchema(1, "M", 25, 99, "10001").validate()


class TestRatingSchema:
    def test_valid_rating(self):
        r = RatingSchema(1, 1, 4, 956703932)
        assert r.validate()

    def test_all_valid_ratings(self):
        for rating in VALID_RATINGS:
            r = RatingSchema(1, 1, rating, 956703932)
            assert r.validate()

    def test_invalid_rating_zero(self):
        with pytest.raises(AssertionError):
            RatingSchema(1, 1, 0, 956703932).validate()

    def test_invalid_rating_six(self):
        with pytest.raises(AssertionError):
            RatingSchema(1, 1, 6, 956703932).validate()


class TestValidateDatFile:
    def test_movies_dat_valid(self):
        result = validate_dat_file(
            str(ROOT / "data/raw/movies.dat"), MovieSchema
        )
        assert result["invalid"] == 0
        assert result["valid"]   >  0

    def test_users_dat_valid(self):
        result = validate_dat_file(
            str(ROOT / "data/raw/users.dat"), UserSchema
        )
        assert result["invalid"] == 0
        assert result["valid"]   >  0

    def test_ratings_dat_valid(self):
        result = validate_dat_file(
            str(ROOT / "data/raw/ratings.dat"), RatingSchema
        )
        assert result["invalid"] == 0
        assert result["valid"]   >  0
