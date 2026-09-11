import streamlit as st
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import joblib

# Configuration de la page
st.set_page_config(page_title="Scoring Crédit", layout="wide")
st.title("📊 Décision d'Octroi de Crédit")

# Chargement optimisé du modèle et des données
@st.cache_resource
def load_resources():
    model = joblib.load("model/lgbm_model.pkl")
    # Chargement d'un échantillon pour la démo
    df = pd.read_csv("data/train_cleaned_final.csv").fillna(0)
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])
    return model, df, X

model, df, X_sample = load_resources()
SEUIL_OPTIMAL = 0.51

# Sidebar pour simuler l'entrée d'un client
st.sidebar.header("Paramètres Client")
client_idx = st.sidebar.number_input("Sélectionnez l'index d'un client", min_value=0, max_value=len(X_sample)-1, value=0)

if st.button("Analyser le dossier"):
    client_data = X_sample.iloc[[client_idx]]
    
    # 1. Prédiction du risque
    proba_defaut = model.predict_proba(client_data)[:, 1][0]
    
    # 2. Décision Métier
    st.subheader("Résultat de la demande")
    if proba_defaut > SEUIL_OPTIMAL:
        st.error(f"Crédit Refusé (Probabilité de défaut : {proba_defaut:.2f})")
    else:
        st.success(f"Crédit Accepté (Probabilité de défaut : {proba_defaut:.2f})")

    # 3. Explicabilité Locale avec SHAP
    st.subheader("Explication de la décision (SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(client_data)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)