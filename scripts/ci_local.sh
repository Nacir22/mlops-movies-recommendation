#!/bin/bash
# scripts/ci_local.sh
# Reproduit le pipeline CI GitHub Actions en local.
# À lancer avant chaque `git push` pour valider que les tests passent.
#
# Usage :
#   bash scripts/ci_local.sh           # Tous les jobs
#   bash scripts/ci_local.sh --lint    # Lint seulement
#   bash scripts/ci_local.sh --unit    # Tests unitaires seulement
#   bash scripts/ci_local.sh --ml      # Tests ML seulement
#   bash scripts/ci_local.sh --api     # Tests API seulement
#   bash scripts/ci_local.sh --all     # Tout (défaut)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ── Couleurs ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; RESET='\033[0m'

# ── Variables ────────────────────────────────────────────────────────────────
JOBS=()
FAILED=()
PASSED=()
T_START=$(date +%s)

# ── Helpers ──────────────────────────────────────────────────────────────────
header() { echo -e "\n${BLUE}${BOLD}══════════════════════════════════════${RESET}"; echo -e "${BLUE}${BOLD}  $1${RESET}"; echo -e "${BLUE}${BOLD}══════════════════════════════════════${RESET}"; }
ok()     { echo -e "  ${GREEN}✅ $1${RESET}"; PASSED+=("$1"); }
fail()   { echo -e "  ${RED}❌ $1${RESET}"; FAILED+=("$1"); }
warn()   { echo -e "  ${YELLOW}⚠️  $1${RESET}"; }
skip()   { echo -e "  ⏭️  $1 (ignoré)"; }

run_job() {
    local name="$1"; shift
    header "$name"
    if "$@"; then
        ok "$name"
    else
        fail "$name"
        return 1
    fi
}

# ── Parse arguments ──────────────────────────────────────────────────────────
MODE="${1:---all}"

# ── Job : Lint ────────────────────────────────────────────────────────────────
job_lint() {
    echo "  Vérification avec ruff..."
    if command -v ruff &>/dev/null; then
        ruff check . \
            --exclude ".venv,data,models,airflow/logs" \
            --select E,W,F,I \
            --ignore E501,W503 \
            --quiet
        echo "  ruff : OK"
    else
        warn "ruff non installé — pip install ruff"
    fi

    echo "  Vérification avec black..."
    if command -v black &>/dev/null; then
        black --check --diff . \
            --exclude "(\.venv|data/|models/|airflow/logs)" \
            --line-length 100 \
            --quiet
        echo "  black : OK"
    else
        warn "black non installé — pip install black"
    fi
}

# ── Job : Tests unitaires ─────────────────────────────────────────────────────
job_unit() {
    if command -v pytest &>/dev/null; then
        pytest tests/unit/ -v --tb=short -q
    else
        python3 -m pytest tests/unit/ -v --tb=short -q
    fi
}

# ── Job : Entraînement ────────────────────────────────────────────────────────
job_train() {
    echo "  Entraînement du modèle..."
    python3 training/train.py \
        --gaussian-var-smoothing 1e-2 \
        --bernoulli-alpha 1.0 \
        --gaussian-weight 0.6 \
        --bernoulli-weight 0.4 \
        --min-accuracy 0.60
}

# ── Job : Tests ML ────────────────────────────────────────────────────────────
job_ml() {
    if command -v pytest &>/dev/null; then
        pytest tests/ml/ -v --tb=short
    else
        python3 -m pytest tests/ml/ -v --tb=short
    fi
}

# ── Job : Tests API ───────────────────────────────────────────────────────────
job_api() {
    python3 serving/fastapi/test_server.py --test
}

# ── Job : Tests d'intégration ─────────────────────────────────────────────────
job_integration() {
    if command -v pytest &>/dev/null; then
        pytest tests/integration/ -v --tb=short
    else
        python3 -m pytest tests/integration/ -v --tb=short
    fi
}

# ── Job : Sécurité rapide ─────────────────────────────────────────────────────
job_security() {
    echo "  Scan de secrets basique..."
    # Chercher des patterns sensibles évidents
    FOUND=0
    if grep -rn "password\s*=\s*['\"][^'\"]" --include="*.py" \
            --exclude-dir=".git" --exclude-dir="tests" . 2>/dev/null \
            | grep -v "env\|getenv\|os\.\|example\|mock" | grep -v "^Binary"; then
        warn "Mots de passe potentiels détectés"
        FOUND=1
    fi

    if command -v bandit &>/dev/null; then
        bandit -r . \
            --exclude ".venv,data,models,airflow/logs,tests" \
            --severity-level high \
            --confidence-level high \
            --quiet \
            || warn "Bandit : problèmes HIGH détectés (voir ci-dessus)"
    else
        warn "bandit non installé — pip install bandit"
    fi

    [ $FOUND -eq 0 ] && echo "  Scan secrets : OK"
}

# ── Exécution ──────────────────────────────────────────────────────────────────
echo -e "${BOLD}🚀 CI Local — MovieLens MLOps${RESET}"
echo "   Mode : $MODE"
echo "   Répertoire : $ROOT"

case "$MODE" in
    --lint)
        run_job "Lint & Format" job_lint || true
        ;;
    --unit)
        run_job "Tests Unitaires" job_unit || true
        ;;
    --ml)
        run_job "Entraînement" job_train || true
        run_job "Tests ML" job_ml || true
        ;;
    --api)
        run_job "Tests API" job_api || true
        ;;
    --security)
        run_job "Sécurité" job_security || true
        ;;
    --all | *)
        run_job "Lint & Format"      job_lint        || true
        run_job "Tests Unitaires"    job_unit        || true
        run_job "Entraînement"       job_train       || true
        run_job "Tests ML"           job_ml          || true
        run_job "Tests API"          job_api         || true
        run_job "Tests Intégration"  job_integration || true
        run_job "Sécurité rapide"    job_security    || true
        ;;
esac

# ── Résumé ────────────────────────────────────────────────────────────────────
T_END=$(date +%s)
ELAPSED=$((T_END - T_START))

echo ""
echo -e "${BOLD}══════════════════════════════════════${RESET}"
echo -e "${BOLD}  Résumé CI Local — ${ELAPSED}s${RESET}"
echo -e "${BOLD}══════════════════════════════════════${RESET}"

for job in "${PASSED[@]:-}"; do
    [ -n "$job" ] && echo -e "  ${GREEN}✅ $job${RESET}"
done

for job in "${FAILED[@]:-}"; do
    [ -n "$job" ] && echo -e "  ${RED}❌ $job${RESET}"
done

if [ ${#FAILED[@]} -eq 0 ]; then
    echo -e "\n${GREEN}${BOLD}✅ Tous les jobs réussis — prêt à pusher !${RESET}"
    exit 0
else
    echo -e "\n${RED}${BOLD}❌ ${#FAILED[@]} job(s) échoué(s) — corriger avant de pusher${RESET}"
    exit 1
fi
