import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. FONCTIONS DE FEATURE ENGINEERING MÉTIER
# ==========================================

def process_application(df):
    """Nettoyage et features de la table principale"""
    print("Traitement de Application...")
    # Anomalies
    df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
    df['DAYS_BIRTH'] = abs(df['DAYS_BIRTH'])
    
    # Ratios financiers
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    return df

def process_bureau(bureau_path):
    """Traitement de l'historique dans les autres banques"""
    print("Traitement de Bureau...")
    bureau = pd.read_csv(bureau_path)
    bureau_ohe = pd.get_dummies(bureau, dummy_na=True)
    
    bureau_agg = bureau_ohe.drop(columns=['SK_ID_BUREAU']).groupby('SK_ID_CURR').agg(['mean', 'max', 'sum'])
    bureau_agg.columns = ['BUREAU_' + '_'.join(col).strip() for col in bureau_agg.columns.values]
    return bureau_agg.reset_index()

def process_previous_application(prev_path):
    """Traitement des anciens crédits Home Credit"""
    print("Traitement de Previous Application...")
    prev = pd.read_csv(prev_path)
    
    # Nouvelle Feature : A-t-il reçu ce qu'il a demandé ?
    prev['APP_CREDIT_PERC'] = prev['AMT_CREDIT'] / prev['AMT_APPLICATION']
    
    prev_ohe = pd.get_dummies(prev, dummy_na=True)
    prev_agg = prev_ohe.drop(columns=['SK_ID_PREV']).groupby('SK_ID_CURR').agg(['mean', 'max', 'sum'])
    prev_agg.columns = ['PREV_' + '_'.join(col).strip() for col in prev_agg.columns.values]
    return prev_agg.reset_index()

def process_installments_payments(inst_path):
    """Traitement du comportement de paiement (Retards et impayés)"""
    print("Traitement de Installments Payments...")
    inst = pd.read_csv(inst_path)
    
    # Nouvelles Features : Retards et paiements partiels
    inst['PAYMENT_PERC'] = inst['AMT_PAYMENT'] / inst['AMT_INSTALMENT']
    inst['PAYMENT_DIFF'] = inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']
    inst['DPD'] = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT'] # Days Past Due (Retard)
    inst['DPD'] = inst['DPD'].apply(lambda x: x if x > 0 else 0) # On ne garde que les retards réels
    
    inst_agg = inst.drop(columns=['SK_ID_PREV', 'NUM_INSTALMENT_VERSION', 'NUM_INSTALMENT_NUMBER']).groupby('SK_ID_CURR').agg(['mean', 'max', 'sum'])
    inst_agg.columns = ['INSTAL_' + '_'.join(col).strip() for col in inst_agg.columns.values]
    return inst_agg.reset_index()

def process_credit_card(cc_path):
    """Traitement des cartes de crédit (Plafonds et retraits)"""
    print("Traitement de Credit Card Balance...")
    cc = pd.read_csv(cc_path)
    
    # Nouvelle Feature : Utilisation du plafond de la carte
    cc['LIMIT_USE'] = cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL']
    
    cc_ohe = pd.get_dummies(cc, dummy_na=True)
    cc_agg = cc_ohe.drop(columns=['SK_ID_PREV']).groupby('SK_ID_CURR').agg(['mean', 'max', 'sum'])
    cc_agg.columns = ['CC_' + '_'.join(col).strip() for col in cc_agg.columns.values]
    return cc_agg.reset_index()


# ==========================================
# 2. PIPELINE D'ENCODAGE ET D'ASSEMBLAGE
# ==========================================

def encode_and_align(train, test):
    """Gère le Label Encoding (2 classes) et le One-Hot Encoding (>2 classes)"""
    print("Encodage des variables catégorielles restantes...")
    le = LabelEncoder()
    for col in train.columns:
        if train[col].dtype == 'object':
            if len(list(train[col].unique())) <= 2:
                train[col] = le.fit_transform(train[col].astype(str))
                test[col] = test[col].astype(str).map(lambda s: '<unknown>' if s not in le.classes_ else s)
                le.classes_ = np.append(le.classes_, '<unknown>')
                test[col] = le.transform(test[col])
                
    train = pd.get_dummies(train)
    test = pd.get_dummies(test)
    
    train_labels = train['TARGET']
    train, test = train.align(test, join='inner', axis=1)
    train['TARGET'] = train_labels
    return train, test

def main():
    print("🚀 DÉMARRAGE DE LA 'SUPER USINE' DE FEATURE ENGINEERING...")
    
    # 1. Base
    df_train = process_application(pd.read_csv('data/application_train.csv'))
    df_test = process_application(pd.read_csv('data/application_test.csv'))
    
    # 2. Ajout progressif des tables (avec libération de la RAM à chaque étape)
    tables = [
        (process_bureau, 'data/bureau.csv'),
        (process_previous_application, 'data/previous_application.csv'),
        (process_installments_payments, 'data/installments_payments.csv'),
        (process_credit_card, 'data/credit_card_balance.csv')
    ]
    
    for func, path in tables:
        if os.path.exists(path):
            agg_df = func(path)
            df_train = df_train.merge(agg_df, on='SK_ID_CURR', how='left')
            df_test = df_test.merge(agg_df, on='SK_ID_CURR', how='left')
            del agg_df
            gc.collect()
            print(f"✔️ {path.split('/')[-1]} fusionnée. Taille actuelle : {df_train.shape}")
        else:
            print(f"⚠️ Fichier {path} introuvable. On passe.")

    # 3. Encodage final
    df_train, df_test = encode_and_align(df_train, df_test)
    
    # 4. Sauvegarde
    print(f"\n💾 Sauvegarde en cours... (Taille finale Train: {df_train.shape})")
    df_train.to_csv('data/train_cleaned_final.csv', index=False)
    df_test.to_csv('data/test_cleaned_final.csv', index=False)
    print("🎉 PIPELINE TERMINÉ ! LE DATASET EST PRÊT POUR LE MODÈLE.")

if __name__ == "__main__":
    main()