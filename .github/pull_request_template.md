## Description

<!-- Décrivez les changements apportés par cette PR -->

## Type de changement

- [ ] 🐛 Correction de bug
- [ ] ✨ Nouvelle fonctionnalité
- [ ] ♻️  Refactoring (sans changement de comportement)
- [ ] 📊 Amélioration du modèle ML
- [ ] 🔧 Modification de configuration / infrastructure
- [ ] 📝 Documentation
- [ ] 🔒 Correction de sécurité

## Changements ML (si applicable)

| Métrique | Avant | Après | Delta |
|---|---|---|---|
| Accuracy | | | |
| F1-score | | | |
| ROC-AUC  | | | |

<!-- Joindre le run MLflow ou les artifacts CI si disponibles -->

## Checklist

### Code
- [ ] Le code suit les conventions du projet (ruff + black)
- [ ] Les tests unitaires passent (`pytest tests/unit/`)
- [ ] Les tests ML passent (`pytest tests/ml/`)
- [ ] Les tests API passent (`pytest tests/api/`)
- [ ] Aucun secret ou credential dans le code

### Modèle ML (si modification du pipeline)
- [ ] Le nouveau modèle respecte `accuracy >= 0.60`
- [ ] La cross-validation est stable (`std <= 0.05`)
- [ ] Les features sont documentées dans `configs/model_config.yaml`
- [ ] Le modèle est enregistré dans MLflow avec les métriques

### Infrastructure (si modification Docker/K8s)
- [ ] Le Dockerfile est multi-stage ou minimal
- [ ] Les variables sensibles passent par des Secrets GitHub / K8s
- [ ] Les manifests K8s ont des resource limits
- [ ] Le healthcheck est configuré

### Documentation
- [ ] Le README est mis à jour si nécessaire
- [ ] Les changements d'API sont documentés (schémas Pydantic)
- [ ] Le CHANGELOG est mis à jour (pour les tags de version)

## Références

<!-- Issues liées : Closes #xxx, Fixes #xxx -->
<!-- Run MLflow : lien vers l'expérience -->
<!-- Run CI : lien vers le workflow GitHub Actions -->
