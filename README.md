# Bank Churners

[![Streamlit App](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://bankchurners.human---think.ing/)

## Executive Summary
Ce dépôt présente une étude complète de churn sur un portefeuille de cartes bancaires. L’objectif est double : expliquer les signaux associés à l’attrition et construire un score de risque exploitable pour des actions de rétention. Le projet est conçu comme un cas d’étude de niveau portfolio, avec une attention explicite portée à la rigueur méthodologique, à l’interprétabilité et à la qualité de communication.

## Business Context
Le cas d’usage porte sur l’attrition de clients en services de cartes de crédit. Une équipe CRM ou fidélisation souhaite comprendre quels profils décrochent, quels signaux faibles doivent être surveillés et comment prioriser la prise de contact sans confondre corrélation et causalité.

## Project Goals
- expliquer les profils et comportements les plus associés au churn ;
- produire un score de risque client utilisable ;
- comparer plusieurs modèles en tenant compte du déséquilibre de classes ;
- documenter un seuil de décision cohérent avec un contexte de rétention ;
- fournir un dashboard lisible à la fois pour un public métier et un public data.

## Repository Structure
```text
.
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── modeling/
│   ├── segmentation/
│   ├── visualization/
│   └── utils/
├── reports/
│   ├── figures/
│   └── markdown/
├── dashboard/
├── models/
├── outputs/
└── tests/
```

## Methodology Overview
Le workflow suit une logique complète :
1. audit et nettoyage des données ;
2. comparaison de deux traitements des modalités `Unknown` ;
3. feature engineering orienté comportement bancaire ;
4. benchmark de modèles supervisés ;
5. calibrage éventuel et choix de seuil ;
6. interprétation globale et locale ;
7. segmentation exploratoire des profils ;
8. mise à disposition d’artefacts pour dashboard et rapport.

## Dataset Notes
- le dataset source est versionné dans `data/raw/BankChurners.csv` ;
- les deux colonnes `Naive_Bayes_Classifier_...` sont supprimées immédiatement ;
- `CLIENTNUM` est conservé comme identifiant mais exclu des variables prédictives ;
- la cible est encodée de manière explicite avec churn = classe positive.

## Tech Stack
Le projet est construit en Python pour garder une chaîne de reproduction simple et cohérente :
- `pandas`, `numpy` pour la préparation des données ;
- `scikit-learn`, `imbalanced-learn`, `xgboost` pour la modélisation ;
- `shap` pour l’explicabilité ;
- `matplotlib`, `seaborn`, `plotly` pour les visualisations ;
- `streamlit` pour le dashboard.

## How To Run
### 1. Create the environment
```bash
conda env create -f environment.yml
conda activate bank-churners
```

### 2. Run the analytical pipeline
```bash
python -m src.project_runner
```

### 3. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

### Deploy on Streamlit Community Cloud
- entrypoint : `dashboard/app.py`
- runtime dependencies for the app : `dashboard/requirements.txt`
- pour reproduire l'environnement de développement, sélectionner **Python 3.13** dans les paramètres avancés du déploiement

### Deploy redirect on Cloudflare Pages
- source repository : `Brainfkt/Bank-Churners`
- production branch : `main`
- build command : empty
- build output directory : `docs`
- custom domain : `https://bankchurners.human---think.ing/`
- redirect target : `https://bank-churners-brainfkt.streamlit.app/`

Les détails de configuration sont documentés dans `DEPLOYMENT.md`.

### 4. Run tests
```bash
pytest
```

## Key Outputs
- `outputs/metrics/` : audit, benchmark, métriques et choix de seuil ;
- `outputs/predictions/` : scores individuels et prédictions test ;
- `outputs/segmentation/` : profils de segments / personas ;
- `reports/figures/` : visuels EDA, performance et explicabilité ;
- `models/final_model.joblib` : modèle final sérialisé.

## Key Insights
Le run initial du pipeline retient `XGBoost` pondéré comme meilleur compromis performance / rappel / lisibilité opérationnelle.

Résultats clés sur le jeu de test :
- PR-AUC : `0.956`
- ROC-AUC : `0.991`
- recall churn : `0.955`
- precision churn : `0.761`
- seuil recommandé : `0.285`

Autres constats robustes :
- la stratégie `Unknown` conservée comme catégorie explicite est retenue, car le traitement “missing + imputation” n’apporte pas de gain matériel en PR-AUC ;
- les signaux les plus prédictifs sont surtout liés à l’inactivité, à l’intensité transactionnelle, au niveau d’utilisation et à la profondeur de relation ;
- la segmentation KMeans est jugée trop faible (`silhouette = 0.167`) pour être sur-vendue comme structure client stable ; le projet bascule donc vers des personas de risque plus honnêtes.

Les principaux enseignements détaillés sont synthétisés dans :
- `reports/markdown/churn_report.md`
- le dashboard Streamlit dans `dashboard/app.py`

## Limitations
- le dataset est observé à un instant donné et ne permet pas d’inférer des causalités ;
- les variables disponibles décrivent surtout des comportements et intensités d’usage, pas toute l’expérience client ;
- la segmentation est explicitement rejetée comme vérité métier si sa qualité interne reste faible.

## Future Work
- enrichir l’étude avec des données temporelles ou événementielles ;
- intégrer une calibration plus fine si le contexte métier l’exige ;
- brancher le scoring sur un flux CRM ou un monitoring portefeuille ;
- compléter l’analyse par des expérimentations de rétention ou des coûts de campagne.
