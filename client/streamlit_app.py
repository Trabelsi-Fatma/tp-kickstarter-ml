import datetime

import pandas as pd
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"
MAIN_CATEGORIES = ["Technology", "Design", "Games", "Fashion", "Food"]

st.set_page_config(page_title="Succès Kickstarter", page_icon="🚀")
st.title("🚀 Mon produit va-t-il réussir sur Kickstarter ?")
st.write("Renseignez la fiche du projet **avant son lancement** : l'API prédit "
         "s'il atteindra son objectif de financement.")


@st.cache_data(ttl=60)
def charger_catalogue():
    r = requests.get(f"{API_URL}/categories", timeout=5)
    r.raise_for_status()
    return r.json()


try:
    catalogue = charger_catalogue()
except requests.exceptions.RequestException:
    st.error("Impossible de joindre l'API. Le serveur est-il lancé ? "
             "(uvicorn app.main:app --reload)")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    nom = st.text_input("Nom de la campagne", value="Awesome Board Game!")
    main_category = st.selectbox("Catégorie principale", MAIN_CATEGORIES, index=2)
    category = st.selectbox("Sous-catégorie", catalogue["categories"][main_category])
    country = st.selectbox("Pays", catalogue["countries"],
                           index=catalogue["countries"].index("US")
                           if "US" in catalogue["countries"] else 0)
with col2:
    objectif = st.number_input("Objectif de financement (USD)", min_value=1.0,
                               value=5000.0, step=500.0)
    duree = st.slider("Durée de la campagne (jours)", 1, 100, 30)
    date_lancement = st.date_input("Date de lancement prévue",
                                   value=datetime.date(2018, 3, 15))

if st.button("Prédire"):
    payload = {
        "nom_projet": nom,
        "main_category": main_category,
        "category": category,
        "country": country,
        "objectif_usd": objectif,
        "duree_jours": duree,
        "date_lancement": date_lancement.isoformat(),
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
    except requests.exceptions.RequestException:
        st.error("Impossible de joindre l'API. Le serveur est-il lancé ?")
    else:
        if response.status_code == 200:
            result = response.json()
            p_ok = result["probabilities"]["reussite"]
            if result["class_name"] == "reussite":
                st.success(f"Résultat prédit : réussite (probabilité {p_ok:.0%})")
            else:
                st.warning(f"Résultat prédit : échec (probabilité de réussite {p_ok:.0%})")
            st.bar_chart(pd.Series(result["probabilities"]))
            st.caption("Le modèle reste modeste (AUC-ROC ≈ 0,76) : le succès dépend aussi "
                       "de facteurs absents des données (qualité, marketing, notoriété).")
        else:
            erreurs = response.json().get("detail", [])
            if response.status_code == 422 and isinstance(erreurs, list):
                for e in erreurs:
                    st.error(f"Champ « {e['loc'][-1]} » invalide : {e['msg']}")
            else:
                st.error(f"Erreur {response.status_code} : {erreurs}")