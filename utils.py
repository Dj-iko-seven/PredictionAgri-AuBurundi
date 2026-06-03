"""Fonctions utilitaires partagees par le pipeline et l'application."""

from __future__ import annotations

import os
import pickle

import pandas as pd

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
MODEL_NAMES = ["decision_tree", "random_forest", "logistic_regression"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=[TARGET])

    for col in FEATURES_NUM + LEAKAGE_COLUMNS:
        if col in df.columns:
            df[col] = df.groupby("culture")[col].transform(lambda s: s.fillna(s.median()))

    return df.fillna(df.median(numeric_only=True))


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_saved_models(model_dir: str = "models") -> tuple[dict, dict]:
    models = {}
    for name in MODEL_NAMES:
        path = os.path.join(model_dir, f"{name}.pkl")
        if os.path.exists(path):
            models[name] = load_pickle(path)

    meta_path = os.path.join(model_dir, "preprocessor.pkl")
    meta = load_pickle(meta_path) if os.path.exists(meta_path) else {}
    return models, meta


def scenario_frame(values: dict, df: pd.DataFrame) -> pd.DataFrame:
    med = df.median(numeric_only=True)
    row = {
        "annee": int(values.get("annee", 2023)),
        "saison": values.get("saison", "A"),
        "province": values.get("province"),
        "culture": values.get("culture"),
        "altitude_m": float(values.get("altitude_m", med.get("altitude_m", 1200))),
        "pluviometrie_mm": float(values.get("pluviometrie_mm", med.get("pluviometrie_mm", 800))),
        "temperature_moy_C": float(values.get("temperature_moy_C", med.get("temperature_moy_C", 21))),
        "superficie_ha": float(values.get("superficie_ha", med.get("superficie_ha", 2))),
        "utilisation_engrais": int(values.get("utilisation_engrais", 0)),
        "acces_irrigation": int(values.get("acces_irrigation", 0)),
        "nb_menages": int(values.get("nb_menages", med.get("nb_menages", 50))),
    }
    return pd.DataFrame([row])
