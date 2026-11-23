from pathlib import Path

import joblib
import pandas as pd

from app.features import build_features

# Même chemin que dans training/train_model.py
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "kickstarter_model.joblib"


def load_model() -> dict:
    """Charge le fichier .joblib : pipeline, encodeur de cible, catalogue, métriques."""
    return joblib.load(MODEL_PATH)


def predict(bundle: dict, features: dict) -> dict:
    """Renvoie la classe prédite et les probabilités pour un projet.

    `features` contient : name, main_category, category, country,
    objectif_usd, duree_jours, date_lancement.
    """
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]

    raw = pd.DataFrame([features])
    x = build_features(raw)                    # mêmes variables qu'à l'entraînement
    probabilities = model.predict_proba(x)[0]
    class_id = int(probabilities.argmax())

    return {
        "class_id": class_id,
        "class_name": str(label_encoder.inverse_transform([class_id])[0]),
        "probabilities": {
            str(name): float(p)
            for name, p in zip(label_encoder.classes_, probabilities)
        },
    }