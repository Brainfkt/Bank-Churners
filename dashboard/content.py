from __future__ import annotations

SECTION_ORDER = [
    "Executive Overview",
    "Customer Profiles",
    "Churn Drivers",
    "Model Performance",
    "Risk Scoring",
    "Customer Segmentation",
    "Business Actions",
]


SECTION_INTROS = {
    "Executive Overview": "Cette vue donne une lecture immédiate du portefeuille filtré : niveau de churn observé, intensité du risque, segments dominants et premiers signaux à surveiller.",
    "Customer Profiles": "Cette section aide à comprendre quels profils client concentrent le plus de risque et comment leurs comportements diffèrent sur les transactions, l'inactivité et la profondeur de relation.",
    "Churn Drivers": "Ici, l'objectif est d'expliquer ce que le modèle utilise le plus pour discriminer les churners potentiels. Ces variables sont des signaux prédictifs, pas des preuves causales.",
    "Model Performance": "Cette vue explique quels modèles ont été comparés, pourquoi le modèle retenu est préféré, et comment lire les principaux indicateurs dans un contexte de rétention.",
    "Risk Scoring": "Le score de churn traduit un niveau de risque individuel. Cette section aide à lire la distribution des scores, à isoler les populations prioritaires et à distinguer prévision et certitude.",
    "Customer Segmentation": "La segmentation est présentée comme une lecture opérationnelle du portefeuille. Dans ce projet, les personas de risque sont privilégiés car les clusters initiaux sont méthodologiquement faibles.",
    "Business Actions": "Cette dernière section transforme les constats analytiques en hypothèses d'action prudentes : qui prioriser, quels signaux suivre et quels types d'intervention tester.",
}


GLOSSARY = {
    "Score de churn": "Probabilité estimée qu'un client appartienne à la classe churn selon le modèle. Ce n'est pas une certitude, mais un niveau de risque.",
    "Recall churn": "Part des churners réellement identifiés par le modèle. Dans ce contexte, un recall élevé limite le nombre de churners manqués.",
    "Précision churn": "Part des clients ciblés comme churners qui churnent effectivement. Elle mesure le risque de contacter des clients qui seraient restés.",
    "PR-AUC": "Indicateur particulièrement utile quand le churn est rare. Plus il est élevé, mieux le modèle classe les churners potentiels sans se contenter de la majorité non churn.",
    "ROC-AUC": "Mesure de séparation globale entre churners et non churners. Elle reste utile, mais elle est moins alignée que la PR-AUC avec un problème déséquilibré.",
    "Seuil recommandé": "Point de coupure utilisé pour transformer un score en alerte opérationnelle. Il reflète un compromis entre recall et précision.",
    "Persona de risque": "Profil de clientèle synthétique construit pour faciliter l'action. Il ne faut pas le lire comme une vérité causale ni comme une segmentation commerciale définitive.",
}


FILTER_LABELS = {
    "genre": "Genre",
    "tranche_age": "Tranche d'âge",
    "categorie_revenu": "Catégorie de revenu",
    "statut_marital": "Statut marital",
    "niveau_etude": "Niveau d'étude",
    "categorie_carte": "Catégorie de carte",
    "tranche_anciennete": "Ancienneté de la relation",
    "profil_produit": "Mono-produit / multi-produit",
    "statut_activite": "Statut d'activité",
    "bande_risque": "Bande de risque",
    "segment_client": "Segment client",
}


FILTER_ORDER = list(FILTER_LABELS.keys())


UI_LABELS = {
    "identifiant_client": "Identifiant client",
    "age_client": "Âge du client",
    "genre": "Genre",
    "nb_personnes_a_charge": "Nombre de personnes à charge",
    "niveau_etude": "Niveau d'étude",
    "statut_marital": "Statut marital",
    "categorie_revenu": "Catégorie de revenu",
    "categorie_carte": "Catégorie de carte",
    "anciennete_mois": "Ancienneté de la relation (mois)",
    "nb_relations_total": "Nombre total de relations / produits",
    "mois_inactifs_12m": "Mois d'inactivité sur 12 mois",
    "nb_contacts_12m": "Nombre de contacts sur 12 mois",
    "limite_credit": "Limite de crédit",
    "encours_renouvelable": "Encours renouvelable utilisé",
    "credit_disponible": "Crédit encore disponible",
    "variation_montant_t4_t1": "Variation du montant entre T4 et T1",
    "montant_transactions_total": "Montant total des transactions",
    "nb_transactions_total": "Nombre total de transactions",
    "variation_nombre_t4_t1": "Variation du nombre de transactions entre T4 et T1",
    "ratio_utilisation_moyen": "Ratio moyen d'utilisation du crédit",
    "churn_observe": "Churn observé",
    "score_churn": "Score de churn",
    "alerte_seuil_recommande": "Alerte au seuil recommandé",
    "score_decile": "Décile de risque",
    "segment_client": "Segment client",
    "tranche_age": "Tranche d'âge",
    "tranche_anciennete": "Ancienneté de la relation",
    "profil_produit": "Profil produit",
    "statut_activite": "Statut d'activité",
    "bande_risque": "Bande de risque",
    "est_dans_test": "Présent dans l'échantillon de test",
    "prediction_resultat": "Issue de prédiction",
}


COLUMN_EXPLANATIONS = {
    "limite_credit": "Montant maximal de crédit autorisé pour le client.",
    "encours_renouvelable": "Part du crédit renouvelable déjà utilisée.",
    "credit_disponible": "Capacité de crédit encore disponible sur la carte.",
    "variation_montant_t4_t1": "Évolution du montant de dépenses entre le premier et le quatrième trimestre.",
    "variation_nombre_t4_t1": "Évolution du nombre de transactions entre le premier et le quatrième trimestre.",
    "score_churn": "Plus le score est élevé, plus le modèle estime que le risque de churn est important.",
    "mois_inactifs_12m": "Nombre de mois durant lesquels le compte a été inactif sur les 12 derniers mois.",
}


VALUE_TRANSLATIONS = {
    "Gender": {
        "F": "Femme",
        "M": "Homme",
    },
    "Education_Level": {
        "Unknown": "Non renseigné",
        "Uneducated": "Sans diplôme",
        "High School": "Lycée",
        "College": "Études supérieures courtes",
        "Graduate": "Diplôme universitaire",
        "Post-Graduate": "Master / troisième cycle",
        "Doctorate": "Doctorat",
    },
    "Marital_Status": {
        "Unknown": "Non renseigné",
        "Single": "Célibataire",
        "Married": "Marié(e)",
        "Divorced": "Divorcé(e)",
    },
    "Income_Category": {
        "Unknown": "Non renseigné",
        "Less than $40K": "Moins de 40 k$",
        "$40K - $60K": "40 à 60 k$",
        "$60K - $80K": "60 à 80 k$",
        "$80K - $120K": "80 à 120 k$",
        "$120K +": "Plus de 120 k$",
    },
    "Card_Category": {
        "Blue": "Carte Blue",
        "Silver": "Carte Silver",
        "Gold": "Carte Gold",
        "Platinum": "Carte Platinum",
    },
    "prediction_outcome": {
        "true_positive": "Churner bien détecté",
        "true_negative": "Client stable correctement ignoré",
        "false_positive": "Fausse alerte",
        "false_negative": "Churner manqué",
    },
}


MODEL_LABELS = {
    "xgboost_weighted": "XGBoost (weighted)",
    "random_forest_plain": "Random Forest (plain)",
    "random_forest_balanced": "Random Forest (class-weighted)",
    "random_forest_smote": "Random Forest (SMOTE)",
    "logistic_plain": "Logistic Regression (plain)",
    "logistic_balanced": "Logistic Regression (class-weighted)",
    "logistic_smote": "Logistic Regression (SMOTE)",
}


MODEL_FAMILY_LABELS = {
    "XGBoost": "XGBoost",
    "Random Forest": "Random Forest",
    "Logistic Regression": "Logistic Regression",
}


IMBALANCE_LABELS = {
    "none": "Aucun rééquilibrage",
    "class_weight": "Pondération des classes",
    "smote": "Sur-échantillonnage SMOTE",
    "scale_pos_weight": "Pondération de la classe churn",
}


PERSONA_LABELS = {
    "Lower-risk active": "Actifs à risque contenu",
    "Mono-product exposed": "Mono-produit exposés",
    "Dormant high-risk": "Dormants à très haut risque",
    "Contacted but fragile": "Très contactés mais fragiles",
}


PERSONA_DESCRIPTIONS = {
    "Actifs à risque contenu": "Clients encore engagés, avec un score moyen faible et une fréquence transactionnelle soutenue.",
    "Mono-produit exposés": "Clients faiblement équipés, plus vulnérables à la concurrence ou à une faible profondeur de relation.",
    "Dormants à très haut risque": "Clients peu engagés récemment, avec un score très élevé et une forte probabilité de churn.",
    "Très contactés mais fragiles": "Clients déjà sollicités plusieurs fois, mais dont le risque reste élevé, signe possible d'une relation sous tension.",
}


PERSONA_ACTIONS = {
    "Actifs à risque contenu": "Surveiller sans sur-solliciter. Privilégier des signaux de veille et des actions légères de fidélisation.",
    "Mono-produit exposés": "Tester des actions d'approfondissement relationnel ou de montée en gamme si l'appétence commerciale est confirmée.",
    "Dormants à très haut risque": "Priorité élevée pour des campagnes de réactivation ciblées, avec un message simple et un timing rapide.",
    "Très contactés mais fragiles": "Revoir le type de contact plutôt que d'intensifier la pression commerciale. Une approche plus qualitative peut être plus pertinente.",
}


RISK_BAND_DESCRIPTIONS = {
    "Faible": "Le modèle détecte peu de signaux de fragilité à ce stade.",
    "Moyen": "Le risque reste modéré, mais certains signaux justifient une veille.",
    "Élevé": "Le client dépasse le seuil opérationnel recommandé et mérite une attention proactive.",
    "Très élevé": "Le risque est très marqué. Ces clients concentrent les signaux de désengagement les plus nets.",
}


FEATURE_LABELS = {
    "Total_Trans_Ct": "Nombre total de transactions",
    "Total_Revolving_Bal": "Encours renouvelable utilisé",
    "transaction_amount_per_txn": "Montant moyen par transaction",
    "Total_Relationship_Count": "Nombre total de relations / produits",
    "declining_count_flag": "Baisse récente du nombre de transactions",
    "tenure_band_early": "Relation récente",
    "Total_Ct_Chng_Q4_Q1": "Variation du nombre de transactions entre T4 et T1",
    "Total_Trans_Amt": "Montant total des transactions",
    "Months_Inactive_12_mon": "Mois d'inactivité sur 12 mois",
    "is_dormant_3m": "Client inactif depuis au moins 3 mois",
    "Gender_F": "Genre : femme",
    "Gender_M": "Genre : homme",
    "Total_Amt_Chng_Q4_Q1": "Variation du montant entre T4 et T1",
    "Credit_Limit": "Limite de crédit",
    "Contacts_Count_12_mon": "Nombre de contacts sur 12 mois",
    "Income_Category_Unknown": "Revenu non renseigné",
    "Customer_Age": "Âge du client",
    "declining_amt_flag": "Baisse récente du montant dépensé",
    "Avg_Open_To_Buy": "Crédit encore disponible",
    "high_contact_low_activity": "Contacts élevés malgré une faible activité",
}


FEATURE_GROUPS = {
    "Nombre total de transactions": "Activité transactionnelle",
    "Encours renouvelable utilisé": "Utilisation du crédit",
    "Montant moyen par transaction": "Activité transactionnelle",
    "Nombre total de relations / produits": "Profondeur de relation",
    "Baisse récente du nombre de transactions": "Évolution récente de l'activité",
    "Relation récente": "Ancienneté de la relation",
    "Variation du nombre de transactions entre T4 et T1": "Évolution récente de l'activité",
    "Montant total des transactions": "Activité transactionnelle",
    "Mois d'inactivité sur 12 mois": "Engagement récent",
    "Client inactif depuis au moins 3 mois": "Engagement récent",
    "Genre : femme": "Profil sociodémographique",
    "Genre : homme": "Profil sociodémographique",
    "Variation du montant entre T4 et T1": "Évolution récente de l'activité",
    "Limite de crédit": "Capacité financière",
    "Nombre de contacts sur 12 mois": "Pression relationnelle",
    "Revenu non renseigné": "Qualité de donnée / profil",
    "Âge du client": "Profil sociodémographique",
    "Baisse récente du montant dépensé": "Évolution récente de l'activité",
    "Crédit encore disponible": "Utilisation du crédit",
    "Contacts élevés malgré une faible activité": "Pression relationnelle",
}


DISPLAY_TABLE_COLUMNS = [
    "identifiant_client",
    "genre",
    "tranche_age",
    "niveau_etude",
    "statut_marital",
    "categorie_revenu",
    "categorie_carte",
    "tranche_anciennete",
    "profil_produit",
    "statut_activite",
    "segment_client",
    "score_churn",
    "bande_risque",
    "alerte_seuil_recommande",
]
