# Home Credit Default Risk — Initiation au MLOps

## Objectif

Prédire le risque de défaut de remboursement de crédit à partir du dataset Home Credit (compétition Kaggle). Ce projet couvre les premières étapes du cycle MLOps : exploration, feature engineering, modélisation, tracking des expériences et optimisation.

## Structure du projet
```
home-credit-risk/
├── data/
│   ├── raw/                        # Données brutes Kaggle
│   └── processed/                  # Données nettoyées et enrichies
├── docs/
│   └── mlflow_screenshots/         # Captures d'écran de l'interface MLflow
├── mlruns/                         # Artifacts MLflow (tracking des expériences)
├── notebooks/
│   ├── 01_eda_cleaning.ipynb       # Exploration et nettoyage des données
│   ├── 02_eda_tables_annexes_feat_eng.ipynb  # Tables annexes et feature engineering
│   ├── 03_modelisation_mlflow.ipynb          # Modélisation et tracking MLflow
│   └── 04_Optimisation_HP_Seuil.ipynb        # Optimisation hyperparamètres et seuil métier
├── src/
│   └── utils.py                    # Fonctions utilitaires
├── .gitignore
├── README.md
└── requirements.txt
```

## Données

Dataset issu de la compétition Kaggle [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk). Les fichiers de données ne sont pas inclus dans le repo (trop volumineux). Pour reproduire le projet, téléchargez-les depuis Kaggle et placez-les dans `data/raw/`.

## Résultats

| Run | Modèle | AUC | Recall | Coût métier |
|-----|--------|-----|--------|-------------|
| 04_LightGBM_native | LightGBM — défaut | 0.7699 | 0.6858 | - |
| 07_LightGBM_optuna | LightGBM — Optuna | 0.7734 | 0.6408 | 155 886 |

Meilleur modèle : LightGBM natif optimisé avec Optuna (50 trials). Seuil métier optimal : 0.52 (ratio coût FN/FP = 10:1).

## Lancer MLflow UI

Depuis la racine du projet :
```bash
mlflow ui --backend-store-uri file:mlruns
```

Puis ouvrir [http://127.0.0.1:5000](http://127.0.0.1:5000) dans le navigateur.

## Installation
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Technologies

- Python 3.11
- LightGBM 4.6, XGBoost 3.2, scikit-learn 1.8
- MLflow 3.10 (tracking et model registry)
- Optuna (optimisation des hyperparamètres)
- imbalanced-learn (SMOTE)
- pandas, numpy, matplotlib, seaborn