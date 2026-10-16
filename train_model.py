"""
train_model.py
Trains the Dual AI Engine for Industrial Predictive Maintenance:
1. Supervised Machine Failure Classifier (XGBoost / Random Forest)
2. Multi-class Failure Mode Diagnostician (TWF, HDF, PWF, OSF)
3. Unsupervised Anomaly Detection (Isolation Forest)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

def engineer_features(df):
    """Computes physical engineering features from raw sensors."""
    X = pd.DataFrame()
    X['Air_temp_K'] = df['Air temperature [K]']
    X['Process_temp_K'] = df['Process temperature [K]']
    X['Rot_speed_rpm'] = df['Rotational speed [rpm]']
    X['Torque_Nm'] = df['Torque [Nm]']
    X['Tool_wear_min'] = df['Tool wear [min]']
    
    # 1. Temperature difference (indicator for Heat Dissipation Failure)
    X['Temp_diff_K'] = X['Process_temp_K'] - X['Air_temp_K']
    
    # 2. Rotational Power in Watts: P = Torque * omega = Torque * (rpm * 2 * pi / 60)
    X['Power_Watts'] = X['Torque_Nm'] * (X['Rot_speed_rpm'] * 2 * np.pi / 60.0)
    
    # 3. Overstrain Factor: Product of Tool Wear and Torque
    X['Overstrain_factor'] = X['Torque_Nm'] * X['Tool_wear_min']
    
    # 4. Temperature-Speed Ratio
    X['Temp_Speed_ratio'] = X['Process_temp_K'] / (X['Rot_speed_rpm'] + 1e-5)
    
    return X

def get_failure_type(row):
    """Maps binary failure flags to a single descriptive multi-class category."""
    if row['TWF'] == 1:
        return 'Tool Wear Failure (TWF)'
    elif row['HDF'] == 1:
        return 'Heat Dissipation Failure (HDF)'
    elif row['PWF'] == 1:
        return 'Power Failure (PWF)'
    elif row['OSF'] == 1:
        return 'Overstrain Failure (OSF)'
    elif row['RNF'] == 1:
        return 'Random Failure (RNF)'
    else:
        return 'No Failure'

def train_and_save_models():
    data_path = 'ai4i2020.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
        
    print("Loading AI4I 2020 dataset...")
    df = pd.read_csv(data_path)
    
    # Prepare features
    X = engineer_features(df)
    y_binary = df['Machine failure']
    
    # Multi-class target for failure diagnosis
    df['Failure_Type'] = df.apply(get_failure_type, axis=1)
    y_multiclass = df['Failure_Type']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Class distribution: {y_binary.value_counts().to_dict()}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    print("\n--- Training Model 1: Binary Machine Failure Predictor (XGBoost) ---")
    binary_model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=15, # balance minority failure class
        random_state=42,
        eval_metric='logloss'
    )
    binary_model.fit(X_train, y_train)
    
    y_pred = binary_model.predict(X_test)
    y_proba = binary_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    print("\n--- Training Model 2: Failure Mode Diagnostician (Random Forest) ---")
    # Train on failure instances and sample of normal
    diagnosis_model = RandomForestClassifier(
        n_estimators=120,
        max_depth=7,
        class_weight='balanced',
        random_state=42
    )
    diagnosis_model.fit(X, y_multiclass)
    
    print("\n--- Training Model 3: Unsupervised Anomaly Detector (Isolation Forest) ---")
    # Fit strictly on normal data to detect unusual drifts
    X_normal = X[y_binary == 0]
    anomaly_detector = IsolationForest(
        n_estimators=100,
        contamination=0.04,
        random_state=42
    )
    anomaly_detector.fit(X_normal)
    
    # Package model metadata
    feature_names = list(X.columns)
    model_bundle = {
        'binary_model': binary_model,
        'diagnosis_model': diagnosis_model,
        'anomaly_detector': anomaly_detector,
        'feature_names': feature_names,
        'metrics': {
            'roc_auc': roc_auc,
            'features': feature_names
        }
    }
    
    os.makedirs('models', exist_ok=True)
    bundle_path = os.path.join('models', 'pdm_models.joblib')
    joblib.dump(model_bundle, bundle_path)
    print(f"\nSuccessfully saved trained model bundle to {bundle_path}!")

if __name__ == '__main__':
    train_and_save_models()
