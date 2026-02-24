"""
tests/integration/test_pipeline.py
Tests d'intégration du pipeline complet (LocalPipelineRunner).

Vérifie que les 3 composants s'enchaînent correctement
de bout en bout : données → features → train → évaluation.
"""
import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


class TestLocalPipelineRunner:
    """Tests d'intégration du LocalPipelineRunner (sans KFP)."""

    @pytest.fixture(scope="class")
    def pipeline_result(self, tmp_path):
        """Exécute le pipeline complet une seule fois."""
        from pipelines.pipeline import LocalPipelineRunner

        runner = LocalPipelineRunner(
            gaussian_var_smoothing=1e-2,
            bernoulli_alpha=1.0,
            gaussian_weight=0.6,
            bernoulli_weight=0.4,
            test_size=0.2,
            random_state=42,
            min_accuracy=0.55,    # Seuil abaissé pour le test d'intégration
            data_dir=str(ROOT / "data/processed"),
            output_dir=str(tmp_path / "models"),
        )
        return runner.run()

    def test_pipeline_status_succeeded(self, pipeline_result):
        assert pipeline_result["status"] == "Succeeded"

    def test_train_metrics_present(self, pipeline_result):
        metrics = pipeline_result["train_metrics"]
        required = ["cv_accuracy_mean", "cv_accuracy_std", "train_accuracy"]
        for key in required:
            assert key in metrics, f"Métrique manquante : {key}"

    def test_eval_metrics_present(self, pipeline_result):
        metrics = pipeline_result["eval_metrics"]
        required = ["test_accuracy", "test_f1", "test_roc_auc"]
        for key in required:
            assert key in metrics, f"Métrique manquante : {key}"

    def test_model_file_created(self, pipeline_result):
        model_path = Path(pipeline_result["model_path"])
        assert model_path.exists(), f"Fichier modèle non créé : {model_path}"

    def test_cv_accuracy_reasonable(self, pipeline_result):
        cv_mean = pipeline_result["train_metrics"]["cv_accuracy_mean"]
        assert cv_mean >= 0.50, f"CV accuracy trop basse : {cv_mean}"

    def test_test_accuracy_reasonable(self, pipeline_result):
        acc = pipeline_result["eval_metrics"]["test_accuracy"]
        assert acc >= 0.50, f"Test accuracy trop basse : {acc}"

    def test_metrics_consistent(self, pipeline_result):
        """Train accuracy et test accuracy ne doivent pas trop diverger."""
        train_acc = pipeline_result["train_metrics"]["train_accuracy"]
        test_acc  = pipeline_result["eval_metrics"]["test_accuracy"]
        gap       = abs(train_acc - test_acc)
        assert gap <= 0.20, (
            f"Gap train/test trop large : {gap:.4f} "
            f"(train={train_acc:.4f}, test={test_acc:.4f})"
        )


class TestPipelineComponentsIndependent:
    """Vérifie que chaque composant peut s'exécuter indépendamment."""

    def test_prepare_data_standalone(self, tmp_path):
        from pipelines.components.prepare_data import prepare_data_local
        result = prepare_data_local(
            test_size=0.2, random_state=42,
            output_dir=str(ROOT / "data/processed"),
        )
        assert result["stats"]["n_train"] > 0
        assert result["stats"]["n_test"]  > 0
        assert result["stats"]["n_features"] >= 10

    def test_train_model_standalone(self, tmp_path):
        from pipelines.components.train_model import train_model_local
        result = train_model_local(
            data_dir=str(ROOT / "data/processed"),
            output_dir=str(tmp_path),
        )
        assert "model_path" in result
        assert Path(result["model_path"]).exists()
        assert result["metrics"]["cv_accuracy_mean"] > 0.40

    def test_evaluate_model_standalone(self, tmp_path):
        from pipelines.components.train_model    import train_model_local
        from pipelines.components.evaluate_model import evaluate_model_local

        # D'abord entraîner
        train_result = train_model_local(
            data_dir=str(ROOT / "data/processed"),
            output_dir=str(tmp_path),
        )
        # Puis évaluer
        eval_result = evaluate_model_local(
            min_accuracy=0.50,
            data_dir=str(ROOT / "data/processed"),
            model_path=train_result["model_path"],
        )
        assert eval_result["test_accuracy"] >= 0.50
        assert eval_result["test_roc_auc"]  >  0.0

    def test_pipeline_threshold_failure(self, tmp_path):
        """Le pipeline doit lever une erreur si accuracy < min_accuracy."""
        from pipelines.components.train_model    import train_model_local
        from pipelines.components.evaluate_model import evaluate_model_local

        train_result = train_model_local(
            data_dir=str(ROOT / "data/processed"),
            output_dir=str(tmp_path),
        )
        with pytest.raises(ValueError, match="test_accuracy"):
            evaluate_model_local(
                min_accuracy=0.99,   # Seuil impossible
                data_dir=str(ROOT / "data/processed"),
                model_path=train_result["model_path"],
            )
