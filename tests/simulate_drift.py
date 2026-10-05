import requests
import random

API_URL = "http://127.0.0.1:8000/predict"

# Simulation de 500 clients avec des revenus anormalement bas (Drift artificiel)
for i in range(500):
    payload = {
        "age": random.randint(25, 60),
        "is_female": random.choice([0, 1]),
        "AMT_INCOME_TOTAL": random.uniform(10000.0, 30000.0), # Revenus volontairement très bas
        "AMT_CREDIT": random.uniform(200000.0, 600000.0),
        "EXT_SOURCE_2": random.uniform(0.1, 0.3), # Scores dégradés
        "EXT_SOURCE_3": random.uniform(0.1, 0.3)
    }
    
    try:
        requests.post(API_URL, json=payload)
    except requests.exceptions.ConnectionError:
        print("L'API n'est pas lancée.")
        break

print("Injection de 500 requêtes altérées terminée. Vérifiez votre fichier api_logs.jsonl.")