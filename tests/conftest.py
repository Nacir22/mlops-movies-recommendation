"""
tests/conftest.py
Fixtures partagées entre tous les modules de tests.
"""
import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    """Configuration globale pytest."""
    config.addinivalue_line(
        "markers", "slow: tests lents (>5s) — exclure avec -m 'not slow'"
    )
    config.addinivalue_line(
        "markers", "integration: tests d'intégration — nécessitent les données"
    )
    config.addinivalue_line(
        "markers", "ml: tests de performance ML"
    )


@pytest.fixture(scope="session", autouse=True)
def ensure_processed_data():
    """
    Fixture de session : s'assure que les données préparées existent.
    Les génère automatiquement si absent.
    """
    proc_dir = ROOT / "data" / "processed"
    if not (proc_dir / "X_train.csv").exists():
        print("\n⚙️  Génération des données de test...")
        from training.data_loader import prepare_and_save
        prepare_and_save()
        print("✅ Données générées")
    yield
