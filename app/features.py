import numpy as np
import pandas as pd

# Variables numériques (mises à l'échelle avec StandardScaler)
NUM_COLS = [
    "duree_jours", "annee", "mois", "jour_semaine",
    "log_objectif", "longueur_nom", "nb_mots_nom",
]
# Variables binaires 0/1 (laissées telles quelles)
BIN_COLS = ["nom_avec_exclamation", "nom_avec_interrogation"]
# Variables nominales (sans ordre) -> One-Hot Encoding
NOM_COLS = ["main_category", "category", "country", "saison"]

FEATURE_COLS = NUM_COLS + BIN_COLS + NOM_COLS

# Colonnes brutes attendues en entrée de build_features
RAW_COLS = [
    "name", "main_category", "category", "country",
    "objectif_usd", "duree_jours", "date_lancement",
]


def saison(mois: int) -> str:
    if mois in (12, 1, 2):
        return "hiver"
    if mois in (3, 4, 5):
        return "printemps"
    if mois in (6, 7, 8):
        return "ete"
    return "automne"


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Transforme les colonnes brutes (RAW_COLS) en variables du modèle."""
    d = pd.DataFrame(index=raw.index)
    date = pd.to_datetime(raw["date_lancement"])
    name = raw["name"].astype(str)

    d["duree_jours"] = raw["duree_jours"].astype(int)
    d["annee"] = date.dt.year
    d["mois"] = date.dt.month
    d["jour_semaine"] = date.dt.dayofweek            # 0 = lundi
    d["log_objectif"] = np.log1p(raw["objectif_usd"].astype(float))
    d["longueur_nom"] = name.str.len()
    d["nb_mots_nom"] = name.str.split().str.len().fillna(0)
    d["nom_avec_exclamation"] = name.str.contains("!", regex=False).astype(int)
    d["nom_avec_interrogation"] = name.str.contains("?", regex=False).astype(int)

    d["main_category"] = raw["main_category"]
    d["category"] = raw["category"]
    d["country"] = raw["country"]
    d["saison"] = d["mois"].apply(saison)
    return d[FEATURE_COLS]