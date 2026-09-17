import warnings
import json
import os
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

warnings.filterwarnings('ignore', category=RuntimeWarning)

print("=== 1. ANALYSE OPÉRATIONNELLE DES LOGS ===")
log_file = "api_logs.jsonl"
logs = []

if os.path.exists(log_file):
    with open(log_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

if logs:
    df_logs = pd.DataFrame(logs)
    avg_latency = df_logs['latency_ms'].mean()
    errors = df_logs[df_logs['status_code'] >= 400]
    error_rate = (len(errors) / len(df_logs)) * 100
    
    print(f"Requêtes totales   : {len(df_logs)}")
    print(f"Latence moyenne    : {avg_latency:.2f} ms")
    print(f"Taux d'erreur HTTP : {error_rate:.2f} %\n")
    
    # Extraction des caractéristiques envoyées au endpoint /predict
    predict_payloads = [log['request_payload'] for log in logs if log.get('path') == '/predict' and log.get('request_payload')]
    df_prod_logs = pd.DataFrame(predict_payloads)
else:
    print("Aucun log trouvé ou fichier vide.\n")
    df_prod_logs = pd.DataFrame()

print("=== 2. ANALYSE DU DATA DRIFT ===")
print("Chargement des données de référence...")
train_df = pd.read_csv("data/train_cleaned_final.csv")

# Sélection des variables stratégiques communes entre l'API et l'entraînement
key_features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
key_features = [col for col in key_features if col in train_df.columns]
reference_data = train_df[key_features].head(500)

# Conditions de simulation pour le PoC
if len(df_prod_logs) >= 2:
    print(f"Intégration de {len(df_prod_logs)} requêtes réelles depuis l'API...")
    common_cols = [col for col in key_features if col in df_prod_logs.columns]
    production_data = df_prod_logs[common_cols]
else:
    print("Moins de 2 requêtes API trouvées. Simulation avec le jeu de test...")
    test_df = pd.read_csv("data/test_cleaned_final.csv")
    production_data = test_df[key_features].head(500)

print("Génération du rapport Evidently...")
drift_report = Report(metrics=[DataDriftPreset()])
my_eval = drift_report.run(reference_data=reference_data, current_data=production_data)

my_eval.save_html("data_drift_report.html")
print("Succès ! Rapport généré sous 'data_drift_report.html'.")