import pandas as pd
import numpy as np
import re
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb

# 1. Config MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("P6_Comparaison_Full_Families")

def clean_and_scale(df, scale=False):
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    new_cols = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in df.columns]
    df.columns = new_cols
    if scale:
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df

def business_cost_score(y_true, y_pred):
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return int((10 * fn) + (1 * fp))

def train_and_log_model(name, model_obj, X, y):
    if mlflow.active_run():
        mlflow.end_run()
        
    print(f"\n---  Entraînement : {name} ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds_proba = np.zeros(X.shape[0])

    with mlflow.start_run(run_name=name):
        # Log des paramètres
        params = model_obj.get_params()
        simple_params = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}
        mlflow.log_params(simple_params)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
            
            if "LightGBM" in name:
                model_obj.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], 
                             callbacks=[lgb.early_stopping(50, verbose=False)])
            else:
                model_obj.fit(X_tr, y_tr)
                
            oof_preds_proba[val_idx] = model_obj.predict_proba(X_va)[:, 1]
            print(f"Fold {fold+1} fini.")

        # Métriques
        oof_preds_bin = (oof_preds_proba >= 0.5).astype(int)
        metrics = {
            "AUC": float(roc_auc_score(y, oof_preds_proba)),
            "F1_Score": float(f1_score(y, oof_preds_bin)),
            "Recall": float(recall_score(y, oof_preds_bin)),
            "Business_Cost": float(business_cost_score(y, oof_preds_bin))
        }
        mlflow.log_artifact("cost_threshold_curve.png")
        print(" Graphique envoyé dans les Artifacts MLflow !")
        mlflow.log_metrics(metrics)
        print(f"{name} enregistré.")

def main():
    train = pd.read_csv('data/train_cleaned_final.csv')
    y = train['TARGET']
    X_raw = train.drop(columns=['TARGET', 'SK_ID_CURR'])

    # --- PREPARATION DES DONNEES ---
    X_scaled = clean_and_scale(X_raw, scale=True)
    X_clean = clean_and_scale(X_raw, scale=False)

    # 1. Logistic Regression (Linéaire)
    train_and_log_model("Logistic_Regression", LogisticRegression(class_weight='balanced', max_iter=500), X_scaled, y)

    # 2. MLP (Deep Learning / Neurones) - Needs Scaling
    # Note: MLP n'a pas de 'class_weight', il sera naturellement moins bon sur le déséquilibre
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=20, random_state=42)
    train_and_log_model("MLP_NeuralNet", mlp, X_scaled, y)

    # 3. Random Forest (Ensemble/Bagging)
    train_and_log_model("Random_Forest", RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', n_jobs=-1), X_clean, y)

    # 4. LightGBM (Ensemble/Boosting)
    train_and_log_model("LightGBM", lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, class_weight='balanced'), X_clean, y)

    # 5. XGBoost (Ensemble/Boosting)
    ratio = (y == 0).sum() / (y == 1).sum()
    train_and_log_model("XGBoost", xgb.XGBClassifier(n_estimators=500, scale_pos_weight=ratio, learning_rate=0.05), X_clean, y)

if __name__ == "__main__":
    main()