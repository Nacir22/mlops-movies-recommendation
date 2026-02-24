"""
serving/fastapi/schemas.py
Schémas Pydantic pour l'API de recommandation MovieLens.
Compatible avec Pydantic v1/v2 et fallback pur Python sans Pydantic.
"""
from __future__ import annotations
from typing import Literal, Optional

try:
    from pydantic import BaseModel, Field
    _PYDANTIC = True

    class UserFeatures(BaseModel):
        gender:          Literal["M", "F"]
        age:             int
        occupation:      int
        user_avg_rating: Optional[float] = None
        user_n_ratings:  Optional[int]   = None
        user_std_rating: Optional[float] = None

        def to_feature_dict(self) -> dict:
            return {
                "Gender_enc":      1 if self.gender == "M" else 0,
                "Age":             self.age,
                "Occupation":      self.occupation,
                "user_avg_rating": self.user_avg_rating or 3.5,
                "user_n_ratings":  self.user_n_ratings  or 20,
                "user_std_rating": self.user_std_rating  or 1.0,
            }

    class RecommendRequest(BaseModel):
        user:              UserFeatures
        top_k:             int       = 10
        exclude_movie_ids: list[int] = []

    class PredictLikeRequest(BaseModel):
        user:     UserFeatures
        movie_id: int

    class MovieRecommendation(BaseModel):
        movie_id: int
        title:    str
        score:    float
        genres:   str

    class RecommendResponse(BaseModel):
        recommendations: list[MovieRecommendation]
        model_version:   str
        latency_ms:      float

    class PredictLikeResponse(BaseModel):
        movie_id:      int
        will_like:     bool
        probability:   float
        model_version: str
        latency_ms:    float

    class HealthResponse(BaseModel):
        status:        str
        model_loaded:  bool
        model_version: str
        uptime_s:      float

    class MovieInfo(BaseModel):
        movie_id: int
        title:    str
        year:     Optional[float]
        genres:   str

    class MoviesResponse(BaseModel):
        movies: list[MovieInfo]
        total:  int
        offset: int
        limit:  int

    class MetricsResponse(BaseModel):
        total_requests:      int
        requests_per_minute: float
        avg_latency_ms:      float
        p99_latency_ms:      float
        model_version:       str

except ImportError:
    _PYDANTIC = False

    class UserFeatures:
        def __init__(self, gender="M", age=25, occupation=4,
                     user_avg_rating=None, user_n_ratings=None, user_std_rating=None):
            assert gender in ("M","F"), f"gender invalide: {gender}"
            assert 1 <= age <= 100,     f"age invalide: {age}"
            assert 0 <= occupation <= 20
            self.gender=gender; self.age=age; self.occupation=occupation
            self.user_avg_rating=user_avg_rating
            self.user_n_ratings=user_n_ratings
            self.user_std_rating=user_std_rating

        def to_feature_dict(self):
            return {
                "Gender_enc":      1 if self.gender=="M" else 0,
                "Age":             self.age,
                "Occupation":      self.occupation,
                "user_avg_rating": self.user_avg_rating or 3.5,
                "user_n_ratings":  self.user_n_ratings  or 20,
                "user_std_rating": self.user_std_rating  or 1.0,
            }

        @classmethod
        def from_dict(cls, d):
            return cls(gender=d.get("gender","M"), age=d.get("age",25),
                       occupation=d.get("occupation",4),
                       user_avg_rating=d.get("user_avg_rating"),
                       user_n_ratings=d.get("user_n_ratings"),
                       user_std_rating=d.get("user_std_rating"))

    class _Base:
        def __init__(self,**kw): [setattr(self,k,v) for k,v in kw.items()]

    class RecommendRequest(_Base): pass
    class PredictLikeRequest(_Base): pass
    class MovieRecommendation(_Base): pass
    class RecommendResponse(_Base): pass
    class PredictLikeResponse(_Base): pass
    class HealthResponse(_Base): pass
    class MovieInfo(_Base): pass
    class MoviesResponse(_Base): pass
    class MetricsResponse(_Base): pass
