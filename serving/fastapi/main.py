"""
serving/fastapi/main.py
API de recommandation MovieLens — FastAPI production-ready.

Endpoints :
  GET  /                          → Info service
  GET  /health                    → Santé + état du modèle
  POST /recommend                 → Top-K recommandations personnalisées
  POST /predict-like              → Prédire si un utilisateur aimera un film
  GET  /movies?limit=&offset=     → Catalogue paginé
  GET  /metrics                   → Métriques JSON
  GET  /metrics/prometheus        → Métriques format Prometheus

Démarrage :
  uvicorn serving.fastapi.main:app --host 0.0.0.0 --port 8000 --reload

Production :
  gunicorn serving.fastapi.main:app -w 4 -k uvicorn.workers.UvicornWorker
"""
from __future__ import annotations

import os
import time
import logging
from contextlib import asynccontextmanager
from collections import deque
from typing import Optional

import numpy as np

try:
    from fastapi import FastAPI, HTTPException, Depends, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import PlainTextResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from serving.fastapi.schemas import (
    RecommendRequest, RecommendResponse, MovieRecommendation,
    PredictLikeRequest, PredictLikeResponse,
    HealthResponse, MoviesResponse, MovieInfo, MetricsResponse,
)
from serving.fastapi.model_loader import ModelBundle, load_bundle, get_bundle

log = logging.getLogger(__name__)

# ── Collecteur de métriques en mémoire ────────────────────────────────────

class MetricsCollector:
    """Collecteur de métriques léger (pas de dépendance Prometheus)."""

    def __init__(self, window_size: int = 1000):
        self._start_time    = time.time()
        self._total_requests = 0
        self._latencies      = deque(maxlen=window_size)
        self._by_endpoint:  dict[str, int] = {}

    def record(self, endpoint: str, latency_ms: float):
        self._total_requests += 1
        self._latencies.append(latency_ms)
        self._by_endpoint[endpoint] = self._by_endpoint.get(endpoint, 0) + 1

    @property
    def uptime_s(self) -> float:
        return time.time() - self._start_time

    @property
    def avg_latency_ms(self) -> float:
        return float(np.mean(self._latencies)) if self._latencies else 0.0

    @property
    def p99_latency_ms(self) -> float:
        return float(np.percentile(list(self._latencies), 99)) if len(self._latencies) >= 10 else 0.0

    @property
    def requests_per_minute(self) -> float:
        uptime_min = self.uptime_s / 60
        return self._total_requests / uptime_min if uptime_min > 0 else 0.0

    def to_prometheus(self, model_version: str) -> str:
        """Génère le texte Prometheus (format exposition)."""
        lines = [
            "# HELP api_requests_total Nombre total de requêtes",
            "# TYPE api_requests_total counter",
            f'api_requests_total {self._total_requests}',
            "",
            "# HELP api_latency_ms_avg Latence moyenne en ms",
            "# TYPE api_latency_ms_avg gauge",
            f'api_latency_ms_avg {self.avg_latency_ms:.2f}',
            "",
            "# HELP api_latency_ms_p99 Latence P99 en ms",
            "# TYPE api_latency_ms_p99 gauge",
            f'api_latency_ms_p99 {self.p99_latency_ms:.2f}',
            "",
            "# HELP api_uptime_seconds Temps de fonctionnement en secondes",
            "# TYPE api_uptime_seconds gauge",
            f'api_uptime_seconds {self.uptime_s:.0f}',
            "",
        ]
        for endpoint, count in self._by_endpoint.items():
            lines.append(
                f'api_requests_by_endpoint{{endpoint="{endpoint}"}} {count}'
            )
        lines += [
            "",
            "# HELP model_version_info Version du modèle actif",
            "# TYPE model_version_info gauge",
            f'model_version_info{{version="{model_version}"}} 1',
        ]
        return "\n".join(lines)


# Singleton metrics
_metrics = MetricsCollector()


# ── Construction de l'application ─────────────────────────────────────────

def create_app() -> "FastAPI":
    """Factory de l'application FastAPI."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI non installé. pip install fastapi uvicorn")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Charge le modèle au démarrage, libère à l'arrêt."""
        log.info("🚀 Démarrage API — chargement du modèle...")
        bundle = load_bundle()
        app.state.bundle = bundle
        log.info("✅ Modèle prêt : %s (%s)", bundle.version, bundle.source)
        yield
        log.info("🛑 Arrêt API")

    app = FastAPI(
        title="MovieLens Recommender API",
        description=(
            "API de recommandation de films basée sur un modèle Naïve Bayes.\n\n"
            "Entraîné sur le dataset MovieLens 1M avec pipeline MLOps complet."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Middleware latence ─────────────────────────────────────────────────
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request

    class LatencyMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            t0       = time.perf_counter()
            response = await call_next(request)
            ms       = (time.perf_counter() - t0) * 1000
            response.headers["X-Latency-Ms"] = f"{ms:.2f}"
            _metrics.record(request.url.path, ms)
            return response

    app.add_middleware(LatencyMiddleware)

    # ── Dépendance modèle ──────────────────────────────────────────────────
    def get_model(request) -> ModelBundle:
        bundle = getattr(request.app.state, "bundle", None)
        if bundle is None or not bundle.is_loaded:
            bundle = get_bundle()
        return bundle

    # ══════════════════════════════════════════════════════════════════════
    # Endpoints
    # ══════════════════════════════════════════════════════════════════════

    @app.get("/", tags=["Info"])
    async def root():
        """Informations sur le service."""
        return {
            "service":     "MovieLens Recommender API",
            "version":     "1.0.0",
            "docs":        "/docs",
            "health":      "/health",
            "recommend":   "POST /recommend",
        }

    @app.get("/health", response_model=HealthResponse, tags=["Ops"])
    async def health(request):
        """Santé du service et état du modèle."""
        bundle = get_model(request)
        return HealthResponse(
            status        = "ok" if bundle.is_loaded else "degraded",
            model_loaded  = bundle.is_loaded,
            model_version = bundle.version,
            uptime_s      = _metrics.uptime_s,
        )

    @app.post("/recommend", response_model=RecommendResponse, tags=["Inférence"])
    async def recommend(req: RecommendRequest, request):
        """
        Génère les top-K recommandations personnalisées pour un utilisateur.

        Le profil utilisateur (gender, age, occupation) est combiné avec
        les features films pour prédire P(Liked) sur chaque film du catalogue.
        """
        t0     = time.perf_counter()
        bundle = get_model(request)

        if not bundle.is_loaded:
            raise HTTPException(status_code=503, detail="Modèle non disponible")

        user_feats = req.user.to_feature_dict()

        recs_df = bundle.recommend(
            user_features    = user_feats,
            top_k            = req.top_k,
            exclude_movie_ids= req.exclude_movie_ids,
        )

        if recs_df.empty:
            raise HTTPException(status_code=404, detail="Aucun film dans le catalogue")

        recommendations = [
            MovieRecommendation(
                movie_id = int(row.MovieID),
                title    = str(row.Title),
                score    = round(float(row.score), 4),
                genres   = str(row.Genres),
            )
            for _, row in recs_df.iterrows()
        ]

        latency_ms = (time.perf_counter() - t0) * 1000
        return RecommendResponse(
            recommendations = recommendations,
            model_version   = bundle.version,
            latency_ms      = round(latency_ms, 2),
        )

    @app.post("/predict-like", response_model=PredictLikeResponse, tags=["Inférence"])
    async def predict_like(req: PredictLikeRequest, request):
        """
        Prédit si un utilisateur aimera un film spécifique.
        Retourne will_like (bool) et probability P(Liked) ∈ [0, 1].
        """
        t0     = time.perf_counter()
        bundle = get_model(request)

        if not bundle.is_loaded:
            raise HTTPException(status_code=503, detail="Modèle non disponible")

        user_feats = req.user.to_feature_dict()

        try:
            will_like, probability = bundle.predict_like(user_feats, req.movie_id)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Erreur de prédiction : {e}")

        latency_ms = (time.perf_counter() - t0) * 1000
        return PredictLikeResponse(
            movie_id      = req.movie_id,
            will_like     = will_like,
            probability   = round(probability, 4),
            model_version = bundle.version,
            latency_ms    = round(latency_ms, 2),
        )

    @app.get("/movies", response_model=MoviesResponse, tags=["Catalogue"])
    async def list_movies(
        request,
        limit:  int = Query(20, ge=1,  le=200),
        offset: int = Query(0,  ge=0),
    ):
        """Catalogue paginé des films disponibles."""
        bundle = get_model(request)

        if bundle.movies_df is None or len(bundle.movies_df) == 0:
            raise HTTPException(status_code=503, detail="Catalogue non disponible")

        df    = bundle.movies_df
        total = len(df)
        page  = df.iloc[offset:offset + limit]

        movies = [
            MovieInfo(
                movie_id = int(row.MovieID),
                title    = str(row.Title),
                year     = float(row.Year) if not (hasattr(row.Year, '__class__') and
                           row.Year.__class__.__name__ == 'float' and
                           str(row.Year) == 'nan') else None,
                genres   = str(row.Genres),
            )
            for _, row in page.iterrows()
        ]
        return MoviesResponse(movies=movies, total=total, offset=offset, limit=limit)

    @app.get("/metrics", response_model=MetricsResponse, tags=["Ops"])
    async def get_metrics(request):
        """Métriques de performance de l'API (format JSON)."""
        bundle = get_model(request)
        return MetricsResponse(
            total_requests      = _metrics._total_requests,
            requests_per_minute = round(_metrics.requests_per_minute, 2),
            avg_latency_ms      = round(_metrics.avg_latency_ms, 2),
            p99_latency_ms      = round(_metrics.p99_latency_ms, 2),
            model_version       = bundle.version,
        )

    @app.get("/metrics/prometheus", response_class=PlainTextResponse, tags=["Ops"])
    async def get_prometheus_metrics(request):
        """Métriques format Prometheus (scraped par prometheus.yml)."""
        bundle = get_model(request)
        return PlainTextResponse(
            content      = _metrics.to_prometheus(bundle.version),
            media_type   = "text/plain; version=0.0.4",
        )

    return app


# Instanciation pour uvicorn
if FASTAPI_AVAILABLE:
    app = create_app()
else:
    app = None
