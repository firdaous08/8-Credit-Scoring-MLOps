import cProfile
import pstats
import io
import time
import joblib
import pandas as pd
import numpy as np
import warnings

# 1. Neutraliser les avertissements de terminal
warnings.filterwarnings("ignore")

# 2. Chargement du modèle de référence
model_path = "model/lgbm_model.pkl"
print(f"--> Chargement du modèle : {model_path}...")
model = joblib.load(model_path)

# Couper les messages verbeux de LightGBM
if hasattr(model, "set_params"):
    try:
        model.set_params(verbosity=-1, verbose=-1)
    except Exception:
        pass

if hasattr(model, "feature_name_"):
    EXPECTED_COLUMNS = list(model.feature_name_)
elif hasattr(model, "feature_names_in_"):
    EXPECTED_COLUMNS = list(model.feature_names_in_)
else:
    EXPECTED_COLUMNS = []

# Payload de test
sample_payload = {
    "age": 35,
    "is_female": 1,
    "AMT_INCOME_TOTAL": 150000.0,
    "AMT_CREDIT": 400000.0,
    "EXT_SOURCE_2": 0.7,
    "EXT_SOURCE_3": 0.6
}

def pipeline_inference_actuel(payload):
    features_dict = {
        "AMT_INCOME_TOTAL": payload["AMT_INCOME_TOTAL"],
        "AMT_CREDIT": payload["AMT_CREDIT"],
        "EXT_SOURCE_2": payload["EXT_SOURCE_2"],
        "EXT_SOURCE_3": payload["EXT_SOURCE_3"],
        "DAYS_BIRTH": -int(payload["age"]) * 365
    }
    
    if payload["is_female"] == 1:
        features_dict["CODE_GENDER_F"] = 1
    else:
        features_dict["CODE_GENDER_M"] = 1

    input_df = pd.DataFrame([features_dict])
    
    if len(EXPECTED_COLUMNS) > 0:
        full_df = pd.DataFrame(0.0, index=[0], columns=EXPECTED_COLUMNS)
        for col in input_df.columns:
            if col in full_df.columns:
                full_df[col] = input_df[col].values
        df_to_predict = full_df
    else:
        df_to_predict = input_df

    proba = float(model.predict_proba(df_to_predict)[:, 1][0])
    return proba

# --- BENCHMARK ---
print("--> Exécution du benchmark de latence (500 requêtes)...")
for _ in range(20):
    _ = pipeline_inference_actuel(sample_payload)

latencies_ms = []
for _ in range(500):
    start = time.perf_counter()
    _ = pipeline_inference_actuel(sample_payload)
    end = time.perf_counter()
    latencies_ms.append((end - start) * 1000)

latencies_arr = np.array(latencies_ms)
mean_lat = latencies_arr.mean()
med_lat = np.median(latencies_arr)
p95_lat = np.percentile(latencies_arr, 95)

# --- PROFILING cProfile ---
print("--> Profiling en cours avec cProfile (200 requêtes)...")
profiler = cProfile.Profile()
profiler.enable()

for _ in range(200):
    _ = pipeline_inference_actuel(sample_payload)

profiler.disable()

# Formatage des résultats dans un fichier texte dédié
s = io.StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
ps.print_stats(12)
profile_summary = s.getvalue()

with open("rapport_profiling_initial.txt", "w", encoding="utf-8") as f:
    f.write("=== RAPPORT INITIAL DE PROFILING ET LATENCE ===\n\n")
    f.write(f"Nombre de variables du modèle : {len(EXPECTED_COLUMNS)}\n")
    f.write(f"Latence Moyenne : {mean_lat:.2f} ms\n")
    f.write(f"Latence Médiane : {med_lat:.2f} ms\n")
    f.write(f"Latence P95     : {p95_lat:.2f} ms\n\n")
    f.write("Détail des fonctions les plus lentes (cProfile) :\n")
    f.write(profile_summary)

print("\n" + "="*50)
print(" RÉSULTATS DU BENCHMARK INITIAL")
print("="*50)
print(f"• Latence Moyenne : {mean_lat:.2f} ms")
print(f"• Latence Médiane : {med_lat:.2f} ms")
print(f"• Latence P95     : {p95_lat:.2f} ms")
print("="*50)
print("--> Le rapport complet et propre a été généré : rapport_profiling_initial.txt\n")