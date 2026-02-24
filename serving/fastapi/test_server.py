"""
serving/fastapi/test_server.py
Serveur HTTP minimal (stdlib uniquement) pour tester la logique de l'API
sans avoir FastAPI / uvicorn installé.

Reproduit exactement les mêmes handlers que main.py :
  POST /recommend
  POST /predict-like
  GET  /health
  GET  /movies
  GET  /metrics

Usage :
    python serving/fastapi/test_server.py           # démarre sur port 8999
    python serving/fastapi/test_server.py --test    # lance les tests intégrés et quitte
"""
from __future__ import annotations

import sys
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from serving.fastapi.model_loader import ModelBundle, load_bundle
from serving.fastapi.schemas      import UserFeatures  # utilisé pour to_feature_dict


# ── Chargement global du modèle ────────────────────────────────────────────
_bundle: ModelBundle | None = None
_start  = time.time()
_req_count = 0
_latencies = []


def ensure_bundle():
    global _bundle
    if _bundle is None:
        _bundle = load_bundle()
    return _bundle


# ── Handler HTTP ───────────────────────────────────────────────────────────

class APIHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass   # silencieux

    # ── Helpers ────────────────────────────────────────────────────────────

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, detail: str):
        self._send_json({"detail": detail}, status)

    def _user_feats(self, user_data: dict) -> dict:
        """Convertit les champs user en features modèle."""
        gender = user_data.get("gender", "M")
        return {
            "Gender_enc":      1 if gender == "M" else 0,
            "Age":             user_data.get("age", 25),
            "Occupation":      user_data.get("occupation", 4),
            "user_avg_rating": user_data.get("user_avg_rating", 3.5),
            "user_n_ratings":  user_data.get("user_n_ratings",  20),
            "user_std_rating": user_data.get("user_std_rating", 1.0),
        }

    # ── GET ────────────────────────────────────────────────────────────────

    def do_GET(self):
        global _req_count
        _req_count += 1
        t0      = time.perf_counter()
        parsed  = urlparse(self.path)
        path    = parsed.path
        params  = parse_qs(parsed.query)

        if path == "/health":
            bundle = ensure_bundle()
            self._send_json({
                "status":        "ok" if bundle.is_loaded else "degraded",
                "model_loaded":  bundle.is_loaded,
                "model_version": bundle.version,
                "uptime_s":      round(time.time() - _start, 1),
            })

        elif path == "/movies":
            bundle = ensure_bundle()
            limit  = int(params.get("limit",  ["20"])[0])
            offset = int(params.get("offset", ["0"])[0])
            df     = bundle.movies_df
            page   = df.iloc[offset:offset + limit]
            movies = [
                {"movie_id": int(r.MovieID), "title": r.Title,
                 "year":     float(r.Year) if str(r.Year) != "nan" else None,
                 "genres":   r.Genres}
                for _, r in page.iterrows()
            ]
            self._send_json({"movies": movies, "total": len(df),
                              "offset": offset, "limit": limit})

        elif path == "/metrics":
            bundle = ensure_bundle()
            avg_ms = sum(_latencies) / len(_latencies) if _latencies else 0
            self._send_json({
                "total_requests":      _req_count,
                "requests_per_minute": round(_req_count / max((time.time()-_start)/60, 1), 2),
                "avg_latency_ms":      round(avg_ms, 2),
                "p99_latency_ms":      0.0,
                "model_version":       bundle.version,
            })

        elif path == "/" or path == "":
            self._send_json({
                "service":   "MovieLens Recommender API (test server)",
                "version":   "1.0.0",
                "health":    "/health",
                "recommend": "POST /recommend",
            })

        else:
            self._send_error(404, f"Endpoint {path} inconnu")

        _latencies.append((time.perf_counter() - t0) * 1000)

    # ── POST ───────────────────────────────────────────────────────────────

    def do_POST(self):
        global _req_count
        _req_count += 1
        t0   = time.perf_counter()
        path = urlparse(self.path).path

        try:
            body = self._read_json()
        except Exception as e:
            self._send_error(400, f"JSON invalide : {e}")
            return

        if path == "/recommend":
            bundle = ensure_bundle()
            user   = body.get("user")
            if not user:
                self._send_error(422, "Champ 'user' obligatoire")
                return
            if user.get("gender") not in ("M", "F"):
                self._send_error(422, f"gender doit être 'M' ou 'F', reçu: {user.get('gender')!r}")
                return
            top_k  = int(body.get("top_k", 10))
            excl   = body.get("exclude_movie_ids", [])

            user_feats = self._user_feats(user)
            recs_df    = bundle.recommend(user_feats, top_k=top_k,
                                          exclude_movie_ids=excl)
            if recs_df.empty:
                self._send_error(404, "Aucun film disponible")
                return

            recs = [{"movie_id": int(r.MovieID), "title": r.Title,
                      "score": round(float(r.score), 4), "genres": r.Genres}
                    for _, r in recs_df.iterrows()]
            ms = (time.perf_counter() - t0) * 1000
            self._send_json({"recommendations": recs,
                              "model_version":   bundle.version,
                              "latency_ms":      round(ms, 2)})

        elif path == "/predict-like":
            bundle   = ensure_bundle()
            user     = body.get("user", {})
            movie_id = int(body.get("movie_id", 1))
            user_feats = self._user_feats(user)

            will_like, prob = bundle.predict_like(user_feats, movie_id)
            ms = (time.perf_counter() - t0) * 1000
            self._send_json({"movie_id": movie_id, "will_like": will_like,
                              "probability": round(prob, 4),
                              "model_version": bundle.version,
                              "latency_ms": round(ms, 2)})

        else:
            self._send_error(404, f"Endpoint POST {path} inconnu")

        _latencies.append((time.perf_counter() - t0) * 1000)


# ── Tests intégrés ─────────────────────────────────────────────────────────

def run_tests(host: str = "127.0.0.1", port: int = 8999):
    """Lance les tests sans dépendance externe."""
    import urllib.request

    base = f"http://{host}:{port}"
    PASS = 0; FAIL = 0

    def check(name, fn):
        nonlocal PASS, FAIL
        try:
            fn()
            print(f"  ✅ {name}")
            PASS += 1
        except Exception as e:
            print(f"  ❌ {name} — {e}")
            FAIL += 1

    def get(path):
        with urllib.request.urlopen(f"{base}{path}", timeout=5) as r:
            return json.loads(r.read())

    def post(path, data):
        body = json.dumps(data).encode()
        req  = urllib.request.Request(
            f"{base}{path}", data=body,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())

    time.sleep(0.5)   # laisser le serveur démarrer
    print("\n🧪 Tests de l'API (test_server)\n" + "─" * 45)

    check("GET /health → status=ok",
          lambda: None if get("/health")["status"] == "ok"
          else (_ for _ in ()).throw(AssertionError()))

    check("GET /health → model_loaded=True",
          lambda: None if get("/health")["model_loaded"]
          else (_ for _ in ()).throw(AssertionError()))

    check("GET /movies → liste non vide",
          lambda: None if len(get("/movies?limit=5")["movies"]) > 0
          else (_ for _ in ()).throw(AssertionError()))

    check("GET /movies pagination",
          lambda: None if get("/movies?limit=5&offset=5")["offset"] == 5
          else (_ for _ in ()).throw(AssertionError()))

    rec = post("/recommend", {"user": {"gender":"M","age":25,"occupation":4}, "top_k": 5})
    check("POST /recommend → 5 recommandations",
          lambda: None if len(rec["recommendations"]) == 5
          else (_ for _ in ()).throw(AssertionError(str(len(rec["recommendations"])))))

    check("POST /recommend → scores dans [0,1]",
          lambda: None if all(0 <= r["score"] <= 1 for r in rec["recommendations"])
          else (_ for _ in ()).throw(AssertionError()))

    check("POST /recommend → triés par score décroissant",
          lambda: None if rec["recommendations"][0]["score"] >= rec["recommendations"][-1]["score"]
          else (_ for _ in ()).throw(AssertionError()))

    check("POST /recommend → latency_ms présent",
          lambda: None if "latency_ms" in rec
          else (_ for _ in ()).throw(AssertionError()))

    pred = post("/predict-like", {"user": {"gender":"F","age":35,"occupation":1}, "movie_id": 1})
    check("POST /predict-like → will_like est bool",
          lambda: None if isinstance(pred["will_like"], bool)
          else (_ for _ in ()).throw(AssertionError()))

    check("POST /predict-like → probability dans [0,1]",
          lambda: None if 0 <= pred["probability"] <= 1
          else (_ for _ in ()).throw(AssertionError()))

    check("GET /metrics → total_requests > 0",
          lambda: None if get("/metrics")["total_requests"] > 0
          else (_ for _ in ()).throw(AssertionError()))

    # Test exclusion de films
    rec_excl = post("/recommend", {
        "user": {"gender":"M","age":25,"occupation":4},
        "top_k": 5,
        "exclude_movie_ids": [1, 2, 3, 4, 5]
    })
    excl_ids = {r["movie_id"] for r in rec_excl["recommendations"]}
    check("POST /recommend → films exclus absents",
          lambda: None if not excl_ids & {1,2,3,4,5}
          else (_ for _ in ()).throw(AssertionError(f"Films exclus présents : {excl_ids & {1,2,3,4,5}}")))

    print(f"\n{'─'*45}")
    print(f"📊 {PASS} ✅  |  {FAIL} ❌")
    return FAIL == 0


# ── Point d'entrée ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8999)
    parser.add_argument("--test", action="store_true", help="Lancer les tests et quitter")
    args = parser.parse_args()

    print(f"🚀 Démarrage du serveur de test sur http://{args.host}:{args.port}")
    print("   Chargement du modèle...")

    server = HTTPServer((args.host, args.port), APIHandler)

    if args.test:
        # Démarrer le serveur dans un thread séparé
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        success = run_tests(args.host, args.port)
        server.shutdown()
        sys.exit(0 if success else 1)
    else:
        print(f"✅ Serveur démarré — Ctrl+C pour arrêter")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nArrêt du serveur")
