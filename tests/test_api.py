import pytest
from fastapi.testclient import TestClient

from app.main import app

PROJET = {
    "nom_projet": "Awesome Board Game!",
    "main_category": "Games",
    "category": "Tabletop Games",
    "country": "US",
    "objectif_usd": 5000,
    "duree_jours": 30,
    "date_lancement": "2018-03-15",
}


@pytest.fixture(scope="module")
def client():
    # Le bloc `with` déclenche le démarrage de l'application,
    # donc le chargement du modèle (fonction lifespan).
    with TestClient(app) as c:
        yield c


def test_root(client):
    assert client.get("/").status_code == 200


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_categories(client):
    r = client.get("/categories")
    assert r.status_code == 200
    assert "Tabletop Games" in r.json()["categories"]["Games"]


def test_metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert 0 <= r.json()["auc_roc"] <= 1


def test_predict_valide(client):
    r = client.post("/predict", json=PROJET)
    assert r.status_code == 200
    body = r.json()
    assert body["class_name"] in ("echec", "reussite")
    assert body["class_id"] in (0, 1)


def test_probabilities_sum_to_one(client):
    p = client.post("/predict", json=PROJET).json()["probabilities"]
    assert abs(sum(p.values()) - 1) < 1e-6


def test_predict_missing_field(client):
    data = {k: v for k, v in PROJET.items() if k != "country"}
    assert client.post("/predict", json=data).status_code == 422


def test_predict_objectif_negatif(client):
    assert client.post("/predict", json={**PROJET, "objectif_usd": -10}).status_code == 422


def test_predict_duree_hors_limites(client):
    assert client.post("/predict", json={**PROJET, "duree_jours": 500}).status_code == 422


def test_predict_pays_inconnu(client):
    assert client.post("/predict", json={**PROJET, "country": "ZZ"}).status_code == 422


def test_predict_categorie_principale_inconnue(client):
    assert client.post("/predict", json={**PROJET, "main_category": "Music"}).status_code == 422


def test_predict_sous_categorie_incoherente(client):
    # « Apparel » appartient à Fashion, pas à Games
    r = client.post("/predict", json={**PROJET, "category": "Apparel"})
    assert r.status_code == 422


def test_predict_date_invalide(client):
    assert client.post("/predict", json={**PROJET, "date_lancement": "pas-une-date"}).status_code == 422