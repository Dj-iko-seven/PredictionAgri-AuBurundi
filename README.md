# 🌱 PredictionAgri-AuBurundi

Application web de prédiction des bonnes et mauvaises récoltes agricoles au Burundi par Machine Learning.

**Université Polytechnique de Gitega — Bac4 Génie Logiciel**
TP IA Appliquée — 2025/2026

---

## 🚀 Lancer l'application localement

### 1. Cloner le dépôt
```bash
git clone https://github.com/Dj-iko-seven/PredictionAgri-AuBurundi.git
cd PredictionAgri-AuBurundi
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Lancer Streamlit
```bash
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501`

---

## 🌐 Application déployée

👉 **URL publique :** `https://predictionagri-auburundi.streamlit.app`

---

## 📁 Structure du projet

```
PredictionAgri-AuBurundi/
├── app.py                          # Application Streamlit principale
├── agriculture_burundi.csv         # Dataset (1620 observations, 14 variables)
├── TP_AgriPredict_Burundi.ipynb    # Notebook Jupyter (Ex. 1-6)
├── Rapport_Q29_Q30.md              # Rapport de réflexion (Q29 & Q30)
├── requirements.txt                # Dépendances Python
└── README.md                       # Ce fichier
```

---

## 📊 Contenu du TP

| Exercice | Contenu | Points |
|---|---|---|
| Ex. 1 | Chargement, exploration, qualité des données | 20 pts |
| Ex. 2 | Prétraitement, encodage, normalisation, split | 15 pts |
| Ex. 3 | Arbre de Décision + analyse overfitting | 20 pts |
| Ex. 4 | Forêt Aléatoire + validation croisée | 20 pts |
| Ex. 5 | Régression Logistique + courbe ROC | 20 pts |
| Ex. 6 | Prédiction sur 4 scénarios réels | 15 pts |
| Ex. 7 | Application web déployée (BONUS) | 15 pts |

---

## 🤖 Modèles ML utilisés
- **Arbre de Décision** (max_depth=4, criterion=gini)
- **Forêt Aléatoire** (n_estimators=100) ← recommandé en production
- **Régression Logistique** (max_iter=1000)

## 📦 Dépendances
```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```
