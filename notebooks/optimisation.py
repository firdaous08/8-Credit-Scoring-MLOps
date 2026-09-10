import pandas as pd
import numpy as np
import re
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import joblib

# Configuration MLflow
mlflow.set_tracking_uri("sqlite:///../mlflow.db")
mlflow.set_experiment("P6_Etape4_Optimisation")

# 1. Chargement et Nettoyage
print("Chargement des données...")
df = pd.read_csv('../data/train_cleaned_final.csv').fillna(0)
df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in df.columns]

y = df['TARGET']
X = df.drop(columns=['TARGET', 'SK_ID_CURR'])

# Découpage des données
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. Définition du coût métier
def custom_business_cost(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return int((10 * fn) + (1 * fp))

# 3. Optimisation des Hyperparamètres avec Optuna
def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'class_weight': 'balanced',
        'n_estimators': 150,
        'n_jobs': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]
    
    # Pour ce modèle précis, on cherche son seuil optimal
    thresholds = np.linspace(0.1, 0.9, 81)
    costs = [custom_business_cost(y_val, preds, t) for t in thresholds]
    
    # Optuna va maintenant chercher à MINIMISER ce coût exact
    return min(costs)

print(" Lancement d'Optuna centré sur le Coût Métier...")
# On change la direction : on veut MINIMISER l'argent perdu
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20) 

# 4. Entraînement du meilleur modèle et recherche du seuil final
best_params = study.best_params
best_model = lgb.LGBMClassifier(**best_params, class_weight='balanced')

with mlflow.start_run(run_name="LightGBM_Opti_Cout_Metier"):
    mlflow.log_params(best_params)
    best_model.fit(X_train, y_train)
    val_probs = best_model.predict_proba(X_val)[:, 1]
    
    thresholds = np.linspace(0.1, 0.9, 81)
    costs = [custom_business_cost(y_val, val_probs, t) for t in thresholds]
    
    optimal_idx = np.argmin(costs)
    opt_threshold = thresholds[optimal_idx]
    min_cost = costs[optimal_idx]
    
    mlflow.log_metric("Optimal_Threshold", opt_threshold)
    mlflow.log_metric("Min_Business_Cost", min_cost)

    # 5. Graphique Coût vs Seuil
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, costs, color='red', lw=2, label='Coût Métier')
    plt.axvline(opt_threshold, color='green', linestyle='--', label=f'Seuil Optimal: {opt_threshold:.2f}')
    plt.title("Évolution du Coût Métier en fonction du Seuil de Décision")
    plt.xlabel("Seuil (Threshold)")
    plt.ylabel("Coût Total (10*FN + 1*FP)")
    plt.legend()
    joblib.dump(best_model, '../model/lgbm_model.pkl')
    print("Modèle sauvegardé dans le dossier model/ !")
    plt.grid(True, alpha=0.3)
    plt.savefig("cost_threshold_curve.png")   
    mlflow.log_artifact("cost_threshold_curve.png")

    #  6. Les Matrices de Confusion Comparatives ---
    y_pred_50 = (val_probs >= 0.5).astype(int)
    y_pred_opt = (val_probs >= opt_threshold).astype(int)

    cm_50 = confusion_matrix(y_val, y_pred_50)
    cm_opt = confusion_matrix(y_val, y_pred_opt)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Matrice par défaut
    sns.heatmap(cm_50, annot=True, fmt='d', cmap='Reds', ax=axes[0], cbar=False)
    axes[0].set_title("Avant Opti : Seuil Par Défaut (0.51)")
    axes[0].set_xlabel("Prédiction Modèle")
    axes[0].set_ylabel("Réalité Client")

    # Matrice optimisée
    sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False)
    axes[1].set_title(f"Après Opti : Seuil Optimal Métier ({opt_threshold:.2f})")
    axes[1].set_xlabel("Prédiction Modèle")
    axes[1].set_ylabel("Réalité Client")

    plt.tight_layout()
    plt.savefig("matrices_confusion_comparaison.png")
    mlflow.log_artifact("matrices_confusion_comparaison.png")
    
    print("Graphiques (Courbe et Matrices) sauvegardés et envoyés sur MLflow.")

print(f" Terminé ! Seuil optimal trouvé : {opt_threshold:.2f} pour un coût de {min_cost}")