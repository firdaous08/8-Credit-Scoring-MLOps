import time
import warnings
import joblib
import numpy as np
import pandas as pd
import onnxruntime as rt
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

warnings.filterwarnings("ignore")

# 1. Chargement du modèle de référence
model_path = "model/lgbm_model.pkl"
print(f"--> Chargement du modèle : {model_path}...")
model = joblib.load(model_path)

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
    raise ValueError("Impossible de récupérer les noms des colonnes attendues.")

num_features = len(EXPECTED_COLUMNS)
print(f"Modèle chargé avec succès ({num_features} variables d'entrée).")

# Dictionnaire de correspondance (Nom Colonne -> Index NumPy) pour éviter Pandas
col_to_idx = {col: i for i, col in enumerate(EXPECTED_COLUMNS)}

# Payload de test
sample_payload = {
    "age": 35,
    "is_female": 1,
    "AMT_INCOME_TOTAL": 150000.0,
    "AMT_CREDIT": 400000.0,
    "EXT_SOURCE_2": 0.7,
    "EXT_SOURCE_3": 0.6
}

# --- A. PIPELINE BASELINE (Pandas) ---
def predict_baseline(payload):
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
    full_df = pd.DataFrame(0.0, index=[0], columns=EXPECTED_COLUMNS)
    for col in input_df.columns:
        if col in full_df.columns:
            full_df[col] = input_df[col].values

    return float(model.predict_proba(full_df)[:, 1][0])

# --- B. PIPELINE OPTIMISÉ 1 : NumPy Direct ---
def payload_to_numpy(payload):
    """Construit directement un vecteur NumPy (1, N) sans instancier de DataFrame."""
    vec = np.zeros((1, num_features), dtype=np.float32)
    
    # Mapping direct
    mapping = {
        "AMT_INCOME_TOTAL": payload["AMT_INCOME_TOTAL"],
        "AMT_CREDIT": payload["AMT_CREDIT"],
        "EXT_SOURCE_2": payload["EXT_SOURCE_2"],
        "EXT_SOURCE_3": payload["EXT_SOURCE_3"],
        "DAYS_BIRTH": -float(payload["age"]) * 365.0,
        "CODE_GENDER_F": 1.0 if payload["is_female"] == 1 else 0.0,
        "CODE_GENDER_M": 0.0 if payload["is_female"] == 1 else 1.0
    }
    
    for col_name, val in mapping.items():
        if col_name in col_to_idx:
            vec[0, col_to_idx[col_name]] = val
            
    return vec

def predict_numpy(payload):
    x_input = payload_to_numpy(payload)
    return float(model.predict_proba(x_input)[:, 1][0])

# --- C. PIPELINE OPTIMISÉ 2 : ONNX Runtime ---
print("\n--> Conversion du modèle LightGBM vers ONNX...")
onnx_model_path = "model/lgbm_model.onnx"
initial_type = [('float_input', FloatTensorType([None, num_features]))]

# Conversion
onnx_model = onnxmltools.convert_lightgbm(model, initial_types=initial_type, target_opset=12)
onnxmltools.utils.save_model(onnx_model, onnx_model_path)
print(f"Modèle ONNX sauvegardé : {onnx_model_path}")

# Initialisation de la session d'inférence ONNX
onnx_session = rt.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
onnx_input_name = onnx_session.get_inputs()[0].name

def predict_onnx(payload):
    x_input = payload_to_numpy(payload)
    # Sortie 1 = probabilités par classe
    res = onnx_session.run(None, {onnx_input_name: x_input})
    # LightGBM ONNX renvoie une liste de dictionnaires pour les probabilités
    proba_class_1 = res[1][0][1]
    return float(proba_class_1)

# --- D. CONTRÔLE DE NON-RÉGRESSION ---
print("\n" + "="*55)
print("1. TEST DE NON-RÉGRESSION (Validation de la précision)")
print("="*55)

p_base = predict_baseline(sample_payload)
p_numpy = predict_numpy(sample_payload)
p_onnx = predict_onnx(sample_payload)

print(f"Probabilité Baseline (Pandas)  : {p_base:.6f}")
print(f"Probabilité NumPy Direct       : {p_numpy:.6f}  (Écart : {abs(p_base - p_numpy):.2e})")
print(f"Probabilité ONNX Runtime       : {p_onnx:.6f}  (Écart : {abs(p_base - p_onnx):.2e})")

assert np.isclose(p_base, p_numpy, atol=1e-5), "Régression détectée sur NumPy !"
assert np.isclose(p_base, p_onnx, atol=1e-4), "Régression détectée sur ONNX !"
print("Non-régression validée : prédictions strictement cohérentes.")

# --- E. BENCHMARK COMPARATIF (500 inférences) ---
print("\n" + "="*55)
print("2. BENCHMARK COMPARATIF DE PERFORMANCE (500 itérations)")
print("="*55)

def benchmark_fn(fn, name):
    # Chauffe
    for _ in range(20):
        _ = fn(sample_payload)
    times = []
    for _ in range(500):
        t0 = time.perf_counter()
        _ = fn(sample_payload)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    arr = np.array(times)
    return arr.mean(), np.median(arr), np.percentile(arr, 95)

mean_b, med_b, p95_b = benchmark_fn(predict_baseline, "Baseline")
mean_np, med_np, p95_np = benchmark_fn(predict_numpy, "NumPy Direct")
mean_ox, med_ox, p95_ox = benchmark_fn(predict_onnx, "ONNX Runtime")

# Affichage des résultats
print(f"{'Approche':<22} | {'Moyenne':<10} | {'Médiane':<10} | {'P95':<10} | {'Accélération':<12}")
print("-" * 72)
print(f"{'Baseline (Pandas)':<22} | {mean_b:.2f} ms   | {med_b:.2f} ms   | {p95_b:.2f} ms   | 1.0x (Réf)")
print(f"{'NumPy Direct':<22} | {mean_np:.2f} ms   | {med_np:.2f} ms   | {p95_np:.2f} ms   | {mean_b / mean_np:.1f}x plus rapide")
print(f"{'ONNX Runtime':<22} | {mean_ox:.2f} ms   | {med_ox:.2f} ms   | {p95_ox:.2f} ms   | {mean_b / mean_ox:.1f}x plus rapide")
print("=" * 72)