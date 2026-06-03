"""
Pipeline ML pour AgriPredict Burundi.

Execution:
    python ml_pipeline.py
"""

from __future__ import annotations

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

DATA_PATH = "agriculture_burundi.csv"
MODEL_DIR = "models"

FEATURES_CAT = ["saison", "province", "culture"]
FEATURES_NUM = [
    "annee",
    "altitude_m",
    "pluviometrie_mm",
    "temperature_moy_C",
    "superficie_ha",
    "utilisation_engrais",
    "acces_irrigation",
    "nb_menages",
]
LEAKAGE_COLUMNS = ["rendement_t_ha", "production_totale_t"]
TARGET = "bonne_recolte"


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset introuvable: {path}")
    return pd.read_csv(path)


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie la cible et impute les colonnes numeriques par mediane de culture."""
    df = df.copy()
    df = df.dropna(subset=[TARGET])

    numeric_cols = FEATURES_NUM + LEAKAGE_COLUMNS
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df.groupby("culture")[col].transform(lambda s: s.fillna(s.median()))

    df = df.fillna(df.median(numeric_only=True))
    return df


def build_preprocessor() -> ColumnTransformer:
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]
    )
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, FEATURES_CAT),
            ("num", num_pipeline, FEATURES_NUM),
        ],
        remainder="drop",
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "decision_tree": Pipeline(
            steps=[
                ("prep", build_preprocessor()),
                ("clf", DecisionTreeClassifier(max_depth=4, criterion="gini", random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("prep", build_preprocessor()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=None,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("prep", build_preprocessor()),
                ("clf", LogisticRegression(max_iter=1500, random_state=42)),
            ]
        ),
    }


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "roc": (fpr, tpr, thresholds),
    }


def main() -> None:
    df_raw = load_dataset()
    df = clean_training_data(df_raw)

    X = df[FEATURES_CAT + FEATURES_NUM]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    models = build_models()
    results = {}

    print("Entrainement des modeles AgriPredict Burundi")
    print(f"Observations utilisees: {len(df)} | Features: {len(FEATURES_CAT + FEATURES_NUM)}")
    print(f"Colonnes exclues pour eviter le data leakage: {', '.join(LEAKAGE_COLUMNS)}\n")

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        metrics["cv_scores"] = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        results[name] = {"model": model, **metrics}

        with open(os.path.join(MODEL_DIR, f"{name}.pkl"), "wb") as f:
            pickle.dump(model, f)

        print(
            f"[{name}] accuracy={metrics['accuracy']:.4f} | "
            f"auc={metrics['auc']:.4f} | cv={np.mean(metrics['cv_scores']):.4f}"
        )

    meta = {name: {k: v for k, v in res.items() if k != "model"} for name, res in results.items()}
    meta.update(
        {
            "feature_names": FEATURES_CAT + FEATURES_NUM,
            "categorical_features": FEATURES_CAT,
            "numeric_features": FEATURES_NUM,
            "target": TARGET,
            "leakage_columns": LEAKAGE_COLUMNS,
            "X_test": X_test,
            "y_test": y_test,
            "df_stats": df.describe(include="all").to_dict(),
            "province_list": sorted(df["province"].dropna().unique().tolist()),
            "culture_list": sorted(df["culture"].dropna().unique().tolist()),
            "class_distribution": y.value_counts().sort_index().to_dict(),
        }
    )

    with open(os.path.join(MODEL_DIR, "preprocessor.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print("\nModeles et metadonnees sauvegardes dans le dossier models/")


if __name__ == "__main__":
    main()
