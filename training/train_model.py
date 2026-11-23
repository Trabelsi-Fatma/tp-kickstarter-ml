import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                      # pour importer le paquet « app »
from app.features import BIN_COLS, NOM_COLS, NUM_COLS, build_features  # noqa: E402

DATA_PATH = ROOT / "data" / "ks-projects-201801.csv"
MODEL_PATH = ROOT / "models" / "kickstarter_model.joblib"

# Produits uniquement (comme dans le notebook)
CATEGORIES_PRODUITS = ["Technology", "Design", "Games", "Fashion", "Food"]
# None = toutes les données ; 30000 = même échantillon que le notebook
TAILLE_ECHANTILLON = 30000


def load_data():
    """Lit le CSV, nettoie, et renvoie (X, y_text, categories, pays)."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {DATA_PATH}\n"
            "Placez ks-projects-201801.csv dans le dossier data/ du projet."
        )
    df = pd.read_csv(DATA_PATH)
    print(f"CSV chargé : {df.shape}")

    df = df.drop_duplicates()
    df = df[df["state"].isin(["successful", "failed"])].copy()       # projets terminés
    df = df[df["main_category"].isin(CATEGORIES_PRODUITS)].copy()    # produits
    df = df.dropna(subset=["name"])                                   # nom vide
    df = df[df["country"].str.len() == 2]                             # pays invalides

    df["launched"] = pd.to_datetime(df["launched"])
    df["deadline"] = pd.to_datetime(df["deadline"])
    df["duree_jours"] = (df["deadline"] - df["launched"]).dt.days
    # dates absurdes (ex. 1970) et durées anormales
    df = df[(df["launched"].dt.year >= 2009) & (df["duree_jours"].between(1, 100))]

    if TAILLE_ECHANTILLON is not None and len(df) > TAILLE_ECHANTILLON:
        df = df.sample(TAILLE_ECHANTILLON, random_state=42)
    print(f"Projets utilisés : {len(df)}")

    raw = pd.DataFrame({
        "name": df["name"],
        "main_category": df["main_category"],
        "category": df["category"],
        "country": df["country"],
        "objectif_usd": df["usd_goal_real"],
        "duree_jours": df["duree_jours"],
        "date_lancement": df["launched"],
    })
    X = build_features(raw)
    # La cible : « pledged » et « backers » ne sont PAS dans X (fuite de données)
    y_text = df["state"].map({"failed": "echec", "successful": "reussite"})

    # Catalogue des catégories (utilisé par l'API pour valider les entrées)
    categories = {
        m: sorted(g["category"].unique().tolist())
        for m, g in df.groupby("main_category")
    }
    return X, y_text, categories, sorted(df["country"].unique().tolist())


def build_pipeline():
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("bin", "passthrough", BIN_COLS),
        # Catégories rares regroupées automatiquement (~200 projets sur 122 728)
        ("nom", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              min_frequency=0.0016, sparse_output=False), NOM_COLS),
    ])
    return Pipeline([
        ("prep", preprocess),
        ("clf", HistGradientBoostingClassifier(random_state=42)),
    ])


def main():
    X, y_text, categories, pays = load_data()

    label_encoder = LabelEncoder()                 # echec -> 0, reussite -> 1
    y = label_encoder.fit_transform(y_text)
    codes = {c: int(i) for c, i in zip(label_encoder.classes_,
                                       label_encoder.transform(label_encoder.classes_))}
    print(f"Cible encodée : {codes} | taux de réussite : {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = build_pipeline()

    # Validation croisée (5 plis) : le prétraitement est refait dans chaque pli
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    s = cross_validate(model, X_train, y_train, cv=cv, n_jobs=-1,
                       scoring={"f1": "f1", "auc": "roc_auc"})
    print(f"Validation croisée : F1 = {s['test_f1'].mean():.3f} | "
          f"AUC-ROC = {s['test_auc'].mean():.3f}")

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "auc_roc": roc_auc_score(y_test, proba),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    }
    metrics = {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}
    print("Jeu de test :", json.dumps(metrics, indent=2))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({
        "model": model,
        "label_encoder": label_encoder,
        "categories": categories,
        "countries": pays,
        "metrics": metrics,
    }, MODEL_PATH)
    print(f"Modèle sauvegardé dans {MODEL_PATH}")


if __name__ == "__main__":
    main()