from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.model import load_model, predict
from app.schemas import PredictionResponse, ProjectFeatures

# Dictionnaire partagé qui contient le modèle chargé en mémoire
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Exécuté UNE fois au démarrage du serveur
    state["model"] = load_model()
    yield
    # Exécuté à l'arrêt du serveur
    state.clear()


app = FastAPI(title="API Succès Kickstarter", version="1.0.0", lifespan=lifespan)


@app.get("/")
def root():
    return {"message": "API Succès Kickstarter opérationnelle"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/categories")
def categories():
    """Catalogue des catégories connues (sert à remplir les listes du client)."""
    bundle = state.get("model")
    if bundle is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {"categories": bundle["categories"], "countries": bundle["countries"]}


@app.get("/metrics")
def metrics():
    """Performances mesurées sur le jeu de test lors de l'entraînement."""
    bundle = state.get("model")
    if bundle is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return bundle["metrics"]


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(features: ProjectFeatures):
    bundle = state.get("model")
    if bundle is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    # Règle métier : la sous-catégorie doit appartenir à la catégorie principale
    valides = bundle["categories"].get(features.main_category, [])
    if features.category not in valides:
        raise HTTPException(
            status_code=422,
            detail=f"Sous-catégorie « {features.category} » inconnue pour "
                   f"{features.main_category}. Valeurs possibles : {valides}",
        )

    try:
        data = features.model_dump()
        data["name"] = data.pop("nom_projet")
        return predict(bundle, data)
    except Exception:
        # On ne renvoie jamais la trace Python au client
        raise HTTPException(status_code=500, detail="Erreur interne de prédiction")