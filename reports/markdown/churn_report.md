# Executive Summary

Cette étude vise à construire un cadre d’analyse et de scoring du churn pour un portefeuille de cartes bancaires. Le projet cherche à répondre à une logique de décision réelle : identifier les clients les plus exposés, expliquer les signaux les plus utiles à la rétention et proposer une lecture prudente des résultats, sans glisser vers des conclusions causales excessives.

# Business Understanding

Le churn de clients en crédit / carte bancaire est un sujet central pour les équipes CRM, marketing relationnel et service client. Dans ce contexte, rater un churner coûte souvent plus cher que contacter trop large, mais une stratégie de rétention trop large dégrade aussi l’efficacité opérationnelle. Le projet adopte donc une logique de compromis : privilégier le rappel des churners tout en encadrant la précision et le seuil de décision.

# Data Audit

Le dataset contient 10 127 lignes et 23 colonnes initiales. Deux colonnes `Naive_Bayes_Classifier_...` sont retirées immédiatement car elles ne constituent pas des variables explicatives légitimes pour une analyse métier et peuvent servir de raccourcis indésirables. `CLIENTNUM` est traité comme identifiant technique, jamais comme variable prédictive.

Le taux de churn est d’environ 16 %, ce qui impose une gestion explicite du déséquilibre de classes. Les variables `Education_Level`, `Marital_Status` et `Income_Category` contiennent des modalités `Unknown`, interprétées comme une forme de missingness métier plutôt que comme un simple bruit.

# Data Preparation

La cible `Attrition_Flag` est encodée explicitement avec `Attrited Customer = 1` et `Existing Customer = 0`, afin que la classe positive soit le churn. Le pipeline compare deux stratégies de traitement des modalités `Unknown` :

- les conserver comme catégories explicites ;
- les transformer en valeurs manquantes, avec indicateurs dédiés puis imputation catégorielle.

Le choix final est arbitré par la PR-AUC moyenne sur validation croisée pour deux familles de modèles, afin de ne pas trancher uniquement sur un argument de convenance.
Sur ce run, la stratégie consistant à conserver `Unknown` comme modalité explicite est retenue : elle obtient une PR-AUC moyenne légèrement supérieure à la stratégie “missing + imputation”, sans gain suffisant pour justifier une complexification supplémentaire.

# Exploratory Data Analysis

L’EDA s’appuie sur une combinaison de distributions, taux de churn par catégories, heatmap de corrélation, violons sur les variables numériques clés et vues bivariées centrées sur l’activité client. La lecture attendue n’est pas “quelle variable cause le churn ?” mais plutôt “quels signaux comportementaux paraissent accompagner un désengagement ?”.

Les variables d’activité, d’inactivité récente, de volume transactionnel, de nombre de contacts et de profondeur de relation doivent être lues comme des proxys de fragilité ou d’engagement. Leur force prédictive peut être élevée sans pour autant démontrer un mécanisme causal.

# Feature Engineering

Le projet ajoute un petit nombre de variables métier volontairement interprétables :

- `is_monoproduct` pour détecter une relation bancaire plus étroite ou au contraire plus fragile ;
- `is_dormant_3m` pour capter l’inactivité récente ;
- `transaction_amount_per_txn` comme intensité moyenne par transaction ;
- `high_contact_low_activity` pour repérer une friction commerciale ou relationnelle ;
- des indicateurs de baisse d’activité entre Q4 et Q1 ;
- des bandes de tenure et d’utilisation.

L’objectif n’est pas de multiplier les features, mais de rendre plus lisibles certains comportements clients.

# Modeling Strategy

Le benchmark couvre :

- une régression logistique comme baseline interprétable ;
- une forêt aléatoire comme ensemble robuste ;
- XGBoost comme challenger boosting.

Le déséquilibre de classes est traité via plusieurs approches selon les modèles : pas de correction, pondération de classe, et SMOTE lorsqu’il est pertinent. La comparaison repose d’abord sur la PR-AUC, puis sur le rappel des churners, puis sur la lisibilité opérationnelle.
Dans l’état actuel des données, `XGBoost` pondéré (`scale_pos_weight`) domine les autres candidats sur la validation avec une PR-AUC de `0.976`, devant les variantes Random Forest et loin devant la baseline logistique.

# Model Evaluation

La stratégie de validation repose sur une séparation stratifiée `train / validation / test`, complétée par une validation croisée stratifiée sur l’entraînement pour le tuning. Le seuil opérationnel n’est pas laissé à 0,50 par défaut : il est choisi sur la base d’un compromis rappel / précision, avec une logique cohérente avec la rétention.

Le projet produit systématiquement :

- matrice de confusion ;
- ROC ;
- Precision-Recall ;
- métriques de rappel, précision, F1, F2, ROC-AUC, PR-AUC ;
- score de churn individuel.

Sur le jeu de test, le modèle final atteint :

- `PR-AUC = 0.956`
- `ROC-AUC = 0.991`
- `recall churn = 0.955`
- `precision churn = 0.761`
- `F2 = 0.909`

Le seuil recommandé est fixé à `0.285`. Ce choix reflète une logique de rétention agressive mais encore disciplinée : il permet de récupérer la quasi-totalité des churners tout en gardant un niveau de précision supérieur à 0,75 sur le jeu de test.

# Explainability

L’explicabilité combine une lecture globale des importances et des vues SHAP globales / locales. Cette couche sert à confronter la logique du modèle à l’intuition métier :

- le modèle apprend-il surtout des signaux d’activité et d’inactivité ?
- identifie-t-il des clients mono-produit ou déjà fragiles ?
- repose-t-il sur des proxys plausibles mais non causaux ?

Les cas locaux sélectionnés incluent typiquement un churner bien détecté, un faux positif et un faux négatif, afin d’éviter une lecture purement moyenne du modèle.

# Customer Segmentation

La segmentation n’est pas utilisée comme décoration méthodologique. Elle cherche à identifier des profils actionnables à partir de variables comportementales et relationnelles. Si la qualité interne des clusters reste faible, le projet le signale explicitement et privilégie des personas de risque plus honnêtes.
Ici, le meilleur score de silhouette obtenu par KMeans reste faible (`0.167`). Le projet documente donc explicitement que les clusters ne sont pas assez solides pour être interprétés comme des segments stables, et bascule vers des personas de risque pragmatiques tels que “Dormant high-risk”, “Contacted but fragile” ou “Mono-product exposed”.

# Dashboard Design

Le dashboard Streamlit adopte une double lecture :

- un niveau métier pour les KPI, les déciles de risque et les segments ;
- un niveau data pour le benchmark, les courbes de performance et les drivers.

L’application lit exclusivement les artefacts sauvegardés par le pipeline afin de rester reproductible et légère.

# Business Recommendations

Les recommandations doivent rester prudentes :

- surveiller l’inactivité et la baisse d’intensité transactionnelle ;
- repérer les clients fortement contactés mais peu engagés ;
- prioriser les clients à haut score et à faible profondeur de relation ;
- distinguer les actions de réengagement léger des cas plus structurellement fragiles.

# Limitations

Les limites principales sont structurelles :

- absence de temporalité fine ;
- pas de variables explicites sur satisfaction, incidents, concurrence ou qualité de service ;
- difficulté à distinguer cause, symptôme et simple corrélation.

Ce projet doit donc être lu comme un cadre de scoring et d’aide à l’investigation, pas comme une preuve définitive sur les causes du churn.

# Next Steps

- enrichir les données avec des historiques plus longs ;
- connecter les scores à un cadre d’action CRM ;
- documenter les coûts opérationnels de différents seuils ;
- tester des approches de calibration ou de monotonicité si un usage production plus strict est envisagé.
