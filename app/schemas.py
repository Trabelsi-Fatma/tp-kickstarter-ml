from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

MAIN_CATEGORIES = Literal["Technology", "Design", "Games", "Fashion", "Food"]
PAYS = Literal[
    "AT", "AU", "BE", "CA", "CH", "DE", "DK", "ES", "FR", "GB", "HK",
    "IE", "IT", "JP", "LU", "MX", "NL", "NO", "NZ", "SE", "SG", "US",
]


class ProjectFeatures(BaseModel):
    """Données envoyées par le client : la fiche d'un projet AVANT son lancement."""

    nom_projet: str = Field(..., min_length=1, max_length=200,
                            description="Nom de la campagne", examples=["Awesome Board Game!"])
    main_category: MAIN_CATEGORIES = Field(..., description="Catégorie principale",
                                           examples=["Games"])
    category: str = Field(..., min_length=1,
                          description="Sous-catégorie (doit appartenir à la catégorie principale)",
                          examples=["Tabletop Games"])
    country: PAYS = Field(..., description="Pays (code à 2 lettres)", examples=["US"])
    objectif_usd: float = Field(..., gt=0, le=100_000_000,
                                description="Objectif de financement en USD", examples=[5000])
    duree_jours: int = Field(..., ge=1, le=100,
                             description="Durée de la campagne (1 à 100 jours)", examples=[30])
    date_lancement: date = Field(..., description="Date de lancement prévue (AAAA-MM-JJ)",
                                 examples=["2018-03-15"])


class PredictionResponse(BaseModel):
    """Réponse envoyée par l'API."""

    class_id: int
    class_name: str
    probabilities: dict[str, float]