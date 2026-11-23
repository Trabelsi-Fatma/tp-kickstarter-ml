# Prédire le succès d'un produit Kickstarter — API de déploiement (démarche du TP5)

Mini-projet Machine Learning – L3 DSI – ISET Sfax
Trabelsi Fatma, Kanoun Farah

Ce dossier prolonge le notebook : le **pipeline final** (Gradient Boosting) est entraîné,
sauvegardé dans un fichier `.joblib`, puis **déployé** derrière une API FastAPI, testé avec
`pytest` et consommé par un client Streamlit — exactement comme au TP5.

```
tp-kickstarter-ml/
├── app/            features.py (variables), model.py (charger/prédire), schemas.py (Pydantic), main.py (API)
├── client/         streamlit_app.py
├── data/           <- placer ks-projects-201801.csv ici
├── models/         kickstarter_model.joblib (créé par l'entraînement)
├── tests/          test_api.py
├── training/       train_model.py
├── pytest.ini
└── requirements.txt
```

## Étapes (PowerShell, depuis la racine du projet)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1. Placer data/ks-projects-201801.csv puis entraîner + sauvegarder le modèle
python training/train_model.py

# 2. Lancer l'API (laisser ce terminal ouvert)
uvicorn app.main:app --reload
#    -> http://127.0.0.1:8000/docs  (Swagger : tester POST /predict)

# 3. Tests automatiques (2e terminal, venv activé)
pytest -v          # 13 tests doivent passer

# 4. Client Streamlit (2e terminal, venv activé)
streamlit run client/streamlit_app.py
```

## Exemple de requête

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body '{"nom_projet":"Awesome Board Game!","main_category":"Games","category":"Tabletop Games","country":"US","objectif_usd":5000,"duree_jours":30,"date_lancement":"2018-03-15"}'
```

Réponse : `{"class_id":..., "class_name":"echec"|"reussite", "probabilities":{...}}`

## Choix de conception

- **Aucune fuite de données** : seules les informations connues avant le lancement sont utilisées
  (`pledged`, `backers`… sont exclues).
- `app/features.py` est partagé entre l'entraînement et l'API : mêmes variables des deux côtés.
- Tout le prétraitement (StandardScaler + One-Hot) est dans le `Pipeline` sauvegardé, avec les
  catégories rares regroupées automatiquement et les modalités inconnues ignorées.
- L'API valide les entrées (Pydantic : objectif > 0, durée 1–100 jours, pays et catégorie
  principale connus, sous-catégorie cohérente avec la catégorie principale) → erreur 422 sinon.
- `GET /categories` fournit le catalogue au client ; `GET /metrics` renvoie les performances de test.

## Limites

Modèle modeste (AUC-ROC ≈ 0,76) : la qualité du produit, la vidéo, le marketing ou la notoriété
du créateur ne sont pas dans les données. Les années d'entraînement vont de 2009 à 2018 : une date
de lancement plus récente est donc une extrapolation.
