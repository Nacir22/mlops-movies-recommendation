"""
tests/unit/test_features.py
Tests unitaires du pipeline de feature engineering.
"""
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from training.data_loader import (
    load_movies, load_users, load_ratings,
    build_feature_matrix, get_feature_columns,
)


@pytest.fixture(scope="module")
def raw_data():
    return load_movies(), load_users(), load_ratings()


class TestLoadMovies:
    def test_returns_dataframe(self, raw_data):
        movies, _, _ = raw_data
        assert isinstance(movies, pd.DataFrame)

    def test_required_columns(self, raw_data):
        movies, _, _ = raw_data
        assert "MovieID" in movies.columns
        assert "Title"   in movies.columns
        assert "Year"    in movies.columns

    def test_year_extracted(self, raw_data):
        movies, _, _ = raw_data
        valid_years = movies["Year"].dropna()
        assert (valid_years >= 1900).all()
        assert (valid_years <= 2030).all()

    def test_genre_columns_binary(self, raw_data):
        movies, _, _ = raw_data
        genre_cols = [c for c in movies.columns
                      if c not in ["MovieID","Title","Genres","Year"]]
        assert len(genre_cols) > 0, "Aucune colonne de genre"
        for col in genre_cols:
            unique_vals = movies[col].unique()
            assert set(unique_vals).issubset({0, 1}), f"{col} n'est pas binaire"

    def test_no_duplicate_movie_ids(self, raw_data):
        movies, _, _ = raw_data
        assert movies["MovieID"].nunique() == len(movies)


class TestLoadUsers:
    def test_gender_encoded(self, raw_data):
        _, users, _ = raw_data
        assert "Gender_enc" in users.columns
        assert set(users["Gender_enc"].unique()).issubset({0, 1})

    def test_valid_ages(self, raw_data):
        _, users, _ = raw_data
        valid_ages = {1, 18, 25, 35, 45, 50, 56}
        assert set(users["Age"].unique()).issubset(valid_ages)

    def test_valid_occupations(self, raw_data):
        _, users, _ = raw_data
        assert users["Occupation"].between(0, 20).all()


class TestLoadRatings:
    def test_liked_column_binary(self, raw_data):
        _, _, ratings = raw_data
        assert "Liked" in ratings.columns
        assert set(ratings["Liked"].unique()).issubset({0, 1})

    def test_ratings_in_range(self, raw_data):
        _, _, ratings = raw_data
        assert ratings["Rating"].between(1, 5).all()

    def test_positive_rate_reasonable(self, raw_data):
        _, _, ratings = raw_data
        pos_rate = ratings["Liked"].mean()
        assert 0.10 <= pos_rate <= 0.90, f"Taux de positifs anormal : {pos_rate:.2%}"


class TestBuildFeatureMatrix:
    def test_output_shape(self, raw_data):
        movies, users, ratings = raw_data
        df = build_feature_matrix(ratings, users, movies)
        assert len(df) == len(ratings)
        assert len(df.columns) > 5

    def test_no_nulls_after_fillna(self, raw_data):
        movies, users, ratings = raw_data
        df = build_feature_matrix(ratings, users, movies)
        assert df.isnull().sum().sum() == 0

    def test_liked_column_present(self, raw_data):
        movies, users, ratings = raw_data
        df = build_feature_matrix(ratings, users, movies)
        assert "Liked" in df.columns

    def test_feature_columns_function(self, raw_data):
        movies, users, ratings = raw_data
        df       = build_feature_matrix(ratings, users, movies)
        feat_cols = get_feature_columns(df)
        excluded  = {"UserID","MovieID","Rating","Liked","Timestamp"}
        assert not set(feat_cols).intersection(excluded), \
            "Des colonnes exclues sont dans feat_cols"
