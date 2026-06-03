# AgriPredict Burundi

Application Streamlit de prédiction des bonnes et mauvaises récoltes agricoles au Burundi.

Projet de TP IA appliquée, Bac4 Génie Logiciel, Université Polytechnique de Gitega.

## Fonctionnalités

- Accueil avec résumé du dataset et performances des modèles.
- Analyse complète du TP dans un seul menu : exploration, prétraitement, arbre de décision, forêt aléatoire, régression logistique, ROC et scénarios.
- Prédiction personnalisée avec comparaison simultanée des trois modèles.
- Modèles déjà entraînés et sauvegardés dans `models/` pour un déploiement direct.

## Installation locale

```bash
pip install -r requirements.txt
```

## Lancement local

```bash
streamlit run app.py
```

Si les modèles doivent être régénérés :

```bash
python ml_pipeline.py
streamlit run app.py
```

## Déploiement Streamlit Cloud

1. Créer un dépôt GitHub.
2. Envoyer tous les fichiers du projet dans le dépôt, y compris `models/` et `agriculture_burundi.csv`.
3. Aller sur `https://share.streamlit.io`.
4. Choisir le dépôt GitHub.
5. Sélectionner `app.py` comme fichier principal.
6. Déployer.

## Structure

- `app.py` : application Streamlit.
- `ml_pipeline.py` : pipeline d'entraînement et sauvegarde des modèles.
- `utils.py` : fonctions partagées.
- `agriculture_burundi.csv` : dataset.
- `models/` : modèles entraînés.
- `.streamlit/config.toml` : configuration du thème Streamlit.
- `requirements.txt` : dépendances Python.
- `runtime.txt` : version Python recommandée pour Streamlit Cloud.
