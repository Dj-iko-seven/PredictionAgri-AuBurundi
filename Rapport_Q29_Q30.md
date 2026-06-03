# Rapport de Réflexion — Questions 29 & 30
**TP IA Appliquée — Prédiction des Récoltes au Burundi**
Université Polytechnique de Gitega — Bac4 Génie Logiciel

---

## Q29 — Scénario Gitega – Haricot (pluviométrie = 430 mm)

### Prédiction des modèles
Les trois modèles (Arbre de Décision, Forêt Aléatoire, Régression Logistique) prédisent une **mauvaise récolte** pour ce scénario, avec une probabilité élevée. Cette prédiction est cohérente avec la réalité agronomique au Burundi.

### Cohérence avec l'agriculture burundaise
Le Haricot (*Phaseolus vulgaris*) est la culture la plus consommée au Burundi. Il nécessite entre **600 et 900 mm** de pluviométrie par saison pour un rendement satisfaisant. Avec seulement **430 mm**, le déficit hydrique est critique, notamment durant la phase de floraison et de formation des gousses. La province de Gitega, en altitude (~1720 m), subit par ailleurs des températures fraîches qui ralentissent la croissance.

### Recommandations concrètes pour l'agriculteur
1. **Irrigation d'appoint** : installer des systèmes de collecte et stockage des eaux pluviales (citernes, bassins de rétention)
2. **Variétés résistantes** : adopter des variétés de haricot tolérantes à la sécheresse (ex. RWR 2245, RWR 719)
3. **Paillage organique** : couvrir le sol de résidus végétaux pour limiter l'évapotranspiration (-30% de perte en eau)
4. **Décalage saisonnier** : reporter la plantation à la saison B si les prévisions météorologiques sont plus favorables
5. **Diversification** : associer le haricot avec le Sorgho (plus résistant à la sécheresse) en culture mixte
6. **Assurance récolte** : souscrire une assurance agricole indexée sur la pluviométrie

---

## Q30 — Recommandation pour le Ministère de l'Agriculture du Burundi

### Modèle recommandé : Forêt Aléatoire

Après comparaison des trois modèles sur les métriques accuracy, AUC, F1-score et validation croisée, la **Forêt Aléatoire** est le modèle à déployer en production pour les raisons suivantes :

| Critère | Arbre de Décision | Forêt Aléatoire | Régression Logistique |
|---|---|---|---|
| Accuracy | Moyen | **Meilleur** | Moyen |
| AUC | Moyen | **Meilleur** | Moyen |
| Stabilité (CV) | Faible | **Élevée** | Moyenne |
| Résistance overfitting | Faible | **Élevée** | Moyenne |
| Interprétabilité | Haute | Moyenne | Haute |

La forêt aléatoire capture les **interactions non-linéaires** entre variables (altitude × pluie × température) et reste robuste aux valeurs aberrantes — caractéristiques essentielles pour des données agricoles réelles.

### Données supplémentaires pour améliorer les prédictions
- **Indices NDVI** (satellites Sentinel-2) : mesure de la santé de la végétation en temps réel
- **Qualité des sols** : pH, teneur en azote (N), phosphore (P), potassium (K)
- **Prévisions météorologiques** à 30, 60 et 90 jours (météo IGEBU)
- **Prix des intrants** : coût des engrais et pesticides au moment de la saison
- **Données socio-économiques** : accès au crédit agricole, distance aux marchés
- **Historique des maladies** : rouille du haricot, mildiou du maïs, etc.

### Limites du modèle
1. **Dataset simulé** : les données agricoles réelles peuvent présenter des distributions différentes
2. **Absence de dimension temporelle** : le modèle ne tient pas compte des séquences climatiques (ex. sécheresse prolongée suivie de pluies intenses)
3. **Biais géographique** : certaines provinces sont sous-représentées, réduisant la précision locale
4. **Pas de données satellitaires** : l'état réel du sol et de la végétation n'est pas capturé
5. **Chocs exogènes** : conflits, prix des intrants, épidémies phytosanitaires ne sont pas modélisés

**Conclusion** : Le modèle est un outil d'aide à la décision, pas un oracle. Il doit être utilisé en complément du jugement des agronomes de terrain, et mis à jour chaque saison avec les nouvelles données collectées.
