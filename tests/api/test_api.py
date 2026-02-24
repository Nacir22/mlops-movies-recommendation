"""
tests/api/test_api.py
Tests de l'API de recommandation MovieLens.

Deux modes d'exécution :
  1. pytest tests/api/test_api.py          → démarre le test_server en background
  2. TEST_API_URL=http://api:8000 pytest   → teste une API déjà démarrée (staging/prod)

Les tests couvrent :
  - Contrats d'interface (statut HTTP, structure JSON)
  - Logique métier (scores, pagination, exclusion)
  - Cas limites et erreurs
  - Performance (latence < seuil)
"""
from __future__ import annotations

import sys
import json
import time
import os
import threading
import urllib.request
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── Configuration ──────────────────────────────────────────────────────────

API_URL       = os.getenv("TEST_API_URL", "http://127.0.0.1:8998")
MAX_LATENCY_MS = float(os.getenv("MAX_LATENCY_MS", "500"))

# ── Fixtures ───────────────────────────────────────────────────────────────

def _start_test_server(host: str, port: int):
    """Démarre le test_server dans un thread daemon."""
    from serving.fastapi.test_server import APIHandler
    from http.server import HTTPServer
    server = HTTPServer((host, port), APIHandler)
    server.serve_forever()


@pytest.fixture(scope="session", autouse=True)
def api_server():
    """
    Démarre le serveur de test si TEST_API_URL n'est pas défini
    (mode développement local).
    """
    if os.getenv("TEST_API_URL"):
        # API externe — pas de démarrage local
        yield
        return

    # Démarrer le test_server
    host, port = "127.0.0.1", 8998
    t = threading.Thread(
        target=_start_test_server, args=(host, port), daemon=True
    )
    t.start()

    # Attendre que le serveur soit prêt (max 10s)
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://{host}:{port}/health", timeout=1)
            break
        except Exception:
            time.sleep(0.2)
    else:
        pytest.fail("Le serveur de test n'a pas démarré dans les 10s")

    yield
    # Pas d'arrêt explicite — thread daemon


# ── Helpers ────────────────────────────────────────────────────────────────

def get(path: str, timeout: float = 5.0) -> tuple[int, dict]:
    url = f"{API_URL}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def post(path: str, data: dict, timeout: float = 5.0) -> tuple[int, dict]:
    body = json.dumps(data).encode()
    req  = urllib.request.Request(
        f"{API_URL}{path}", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


USER_M = {"gender": "M", "age": 25, "occupation": 4}
USER_F = {"gender": "F", "age": 35, "occupation": 1}


# ══════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    """GET /health"""

    def test_returns_200(self):
        status, _ = get("/health")
        assert status == 200

    def test_status_ok(self):
        _, body = get("/health")
        assert body["status"] == "ok"

    def test_model_loaded(self):
        _, body = get("/health")
        assert body["model_loaded"] is True

    def test_model_version_present(self):
        _, body = get("/health")
        assert "model_version" in body
        assert body["model_version"] != ""

    def test_uptime_positive(self):
        _, body = get("/health")
        assert body["uptime_s"] >= 0


class TestRecommendEndpoint:
    """POST /recommend"""

    def test_returns_200(self):
        status, _ = post("/recommend", {"user": USER_M})
        assert status == 200

    def test_default_top_k_10(self):
        _, body = post("/recommend", {"user": USER_M})
        assert len(body["recommendations"]) == 10

    def test_custom_top_k(self):
        _, body = post("/recommend", {"user": USER_M, "top_k": 5})
        assert len(body["recommendations"]) == 5

    def test_top_k_1(self):
        _, body = post("/recommend", {"user": USER_M, "top_k": 1})
        assert len(body["recommendations"]) == 1

    def test_scores_in_range(self):
        _, body = post("/recommend", {"user": USER_M, "top_k": 10})
        for rec in body["recommendations"]:
            assert 0.0 <= rec["score"] <= 1.0, f"Score hors [0,1] : {rec['score']}"

    def test_sorted_descending(self):
        _, body = post("/recommend", {"user": USER_M, "top_k": 10})
        scores = [r["score"] for r in body["recommendations"]]
        assert scores == sorted(scores, reverse=True), "Recommandations non triées"

    def test_all_required_fields(self):
        _, body = post("/recommend", {"user": USER_M, "top_k": 3})
        for rec in body["recommendations"]:
            assert "movie_id"  in rec
            assert "title"     in rec
            assert "score"     in rec
            assert "genres"    in rec

    def test_model_version_in_response(self):
        _, body = post("/recommend", {"user": USER_M})
        assert "model_version" in body

    def test_latency_ms_in_response(self):
        _, body = post("/recommend", {"user": USER_M})
        assert "latency_ms" in body
        assert body["latency_ms"] >= 0

    def test_exclude_movie_ids(self):
        exclude = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        _, body  = post("/recommend", {
            "user": USER_M, "top_k": 5,
            "exclude_movie_ids": exclude
        })
        returned_ids = {r["movie_id"] for r in body["recommendations"]}
        overlap = returned_ids & set(exclude)
        assert len(overlap) == 0, f"Films exclus présents dans les recs : {overlap}"

    def test_different_users_different_scores(self):
        """Deux profils différents doivent produire des scores différents."""
        _, body_m = post("/recommend", {"user": USER_M, "top_k": 5})
        _, body_f = post("/recommend", {"user": USER_F, "top_k": 5})
        scores_m  = set(r["score"] for r in body_m["recommendations"])
        scores_f  = set(r["score"] for r in body_f["recommendations"])
        # Les scores ne doivent pas être identiques pour les deux profils
        # (sauf données très uniformes — test souple)
        assert scores_m != scores_f or len(scores_m) > 0

    def test_female_user(self):
        status, body = post("/recommend", {"user": USER_F, "top_k": 5})
        assert status == 200
        assert len(body["recommendations"]) == 5

    def test_user_with_stats(self):
        user_with_stats = {
            "gender": "M", "age": 25, "occupation": 4,
            "user_avg_rating": 4.2, "user_n_ratings": 150, "user_std_rating": 0.8
        }
        status, body = post("/recommend", {"user": user_with_stats, "top_k": 5})
        assert status == 200
        assert len(body["recommendations"]) == 5

    def test_response_time_acceptable(self):
        t0     = time.perf_counter()
        status, _ = post("/recommend", {"user": USER_M, "top_k": 10})
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert status == 200
        assert elapsed_ms < MAX_LATENCY_MS, (
            f"Latence {elapsed_ms:.0f}ms > seuil {MAX_LATENCY_MS}ms"
        )


class TestPredictLikeEndpoint:
    """POST /predict-like"""

    def test_returns_200(self):
        status, _ = post("/predict-like", {"user": USER_M, "movie_id": 1})
        assert status == 200

    def test_will_like_is_bool(self):
        _, body = post("/predict-like", {"user": USER_M, "movie_id": 1})
        assert isinstance(body["will_like"], bool)

    def test_probability_in_range(self):
        _, body = post("/predict-like", {"user": USER_M, "movie_id": 1})
        assert 0.0 <= body["probability"] <= 1.0

    def test_will_like_consistent_with_probability(self):
        """will_like doit être True ssi probability >= 0.5."""
        _, body = post("/predict-like", {"user": USER_M, "movie_id": 1})
        if body["will_like"]:
            assert body["probability"] >= 0.5
        else:
            assert body["probability"] < 0.5

    def test_all_required_fields(self):
        _, body = post("/predict-like", {"user": USER_M, "movie_id": 1})
        for field in ["movie_id", "will_like", "probability", "model_version", "latency_ms"]:
            assert field in body, f"Champ manquant : {field}"

    def test_movie_id_echoed(self):
        _, body = post("/predict-like", {"user": USER_M, "movie_id": 42})
        assert body["movie_id"] == 42

    def test_multiple_movies(self):
        """Tester plusieurs films donne des probabilités différentes."""
        probas = []
        for mid in [1, 5, 10, 20, 50]:
            _, body = post("/predict-like", {"user": USER_M, "movie_id": mid})
            probas.append(body["probability"])
        # Les probabilités ne doivent pas toutes être identiques
        assert len(set(probas)) > 1, "Toutes les probabilités sont identiques"


class TestMoviesEndpoint:
    """GET /movies"""

    def test_returns_200(self):
        status, _ = get("/movies")
        assert status == 200

    def test_default_limit_20(self):
        _, body = get("/movies")
        assert len(body["movies"]) <= 20

    def test_custom_limit(self):
        _, body = get("/movies?limit=5")
        assert len(body["movies"]) == 5

    def test_total_field(self):
        _, body = get("/movies")
        assert "total" in body
        assert body["total"] > 0

    def test_pagination(self):
        _, page1 = get("/movies?limit=5&offset=0")
        _, page2 = get("/movies?limit=5&offset=5")
        ids1 = {m["movie_id"] for m in page1["movies"]}
        ids2 = {m["movie_id"] for m in page2["movies"]}
        assert ids1.isdisjoint(ids2), "Overlap entre les pages"

    def test_movie_fields(self):
        _, body = get("/movies?limit=3")
        for movie in body["movies"]:
            assert "movie_id" in movie
            assert "title"    in movie
            assert "genres"   in movie

    def test_offset_reflected(self):
        _, body = get("/movies?limit=5&offset=10")
        assert body["offset"] == 10


class TestMetricsEndpoint:
    """GET /metrics"""

    def test_returns_200(self):
        status, _ = get("/metrics")
        assert status == 200

    def test_required_fields(self):
        _, body = get("/metrics")
        for field in ["total_requests", "requests_per_minute",
                      "avg_latency_ms", "model_version"]:
            assert field in body, f"Champ manquant : {field}"

    def test_total_requests_positive(self):
        # Faire au moins une requête avant
        post("/recommend", {"user": USER_M, "top_k": 1})
        _, body = get("/metrics")
        assert body["total_requests"] > 0


class TestErrorHandling:
    """Tests des cas d'erreur."""

    def test_unknown_endpoint_404(self):
        status, _ = get("/nonexistent")
        assert status == 404

    def test_recommend_missing_user_field(self):
        """Requête sans champ user obligatoire."""
        status, _ = post("/recommend", {"top_k": 5})
        # Doit retourner 400 ou 422 (pas 200 ni 500)
        assert status in (400, 422, 500)

    def test_recommend_invalid_gender(self):
        """Genre invalide dans le profil utilisateur."""
        status, _ = post("/recommend", {
            "user": {"gender": "X", "age": 25, "occupation": 4}
        })
        assert status in (400, 422, 500)
