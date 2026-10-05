import os
import pandas as pd
import requests
import numpy as np

API_URL = "http://127.0.0.1:8000/predict"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
ref_data_path = os.path.join(PROJECT_ROOT, "data", "train_cleaned_final.csv")

df_ref = pd.read_csv(ref_data_path)
sample_data = df_ref.sample(n=100, random_state=42)

print("Envoi d'un trafic nominal basé sur les vraies données...")

for _, row in sample_data.iterrows():
    # Gestion des valeurs NaN pour éviter le crash JSON
    ext_2 = row.get("EXT_SOURCE_2", 0.5)
    ext_3 = row.get("EXT_SOURCE_3", 0.5)
    
    if pd.isna(ext_2): ext_2 = 0.5
    if pd.isna(ext_3): ext_3 = 0.5

    payload = {
        "age": int(abs(row.get("DAYS_BIRTH", -36500) / 365)),
        "is_female": int(row.get("CODE_GENDER_F", 1)) if not pd.isna(row.get("CODE_GENDER_F")) else 1,
        "AMT_INCOME_TOTAL": float(row.get("AMT_INCOME_TOTAL", 150000.0)),
        "AMT_CREDIT": float(row.get("AMT_CREDIT", 500000.0)),
        "EXT_SOURCE_2": float(ext_2),
        "EXT_SOURCE_3": float(ext_3)
    }
    
    try:
        requests.post(API_URL, json=payload)
    except requests.exceptions.ConnectionError:
        print("L'API n'est pas accessible.")
        break

print("Trafic nominal injecté avec succès !")