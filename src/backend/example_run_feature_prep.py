# example_run_feature_prep.py

import pandas as pd
import joblib
from feature_prep import prepare_features

# ===============================
# 1. Load your datasets
# ===============================
# Replace with actual paths to your CSVs
train_real = pd.read_csv('data/raw/train_real.csv')
train_synthetic = pd.read_csv('data/raw/train_synthetic.csv')
val_data = pd.read_csv('data/raw/validation.csv')

# ===============================
# 2. Prepare training features
# ===============================
train_features, artifacts = prepare_features(
    df_real=train_real,
    df_synthetic=train_synthetic,
    fit=True  # fit transformers on training data
)

print("Training features prepared:", train_features.shape)

# ===============================
# 3. Save artifacts (optional, already saved inside feature_prep.py)
# ===============================
# joblib.dump(artifacts, 'src/backend/feature_engineering/train_artifacts.joblib')

# ===============================
# 4. Prepare validation features
# ===============================
# Load saved artifacts
encoders = joblib.load('src/backend/feature_engineering/encoders/label_encoders.joblib')
scaler = joblib.load('src/backend/feature_engineering/scalers/scaler.joblib')
power_transformer = joblib.load('src/backend/feature_engineering/scalers/power_transformer.joblib')

val_features, _ = prepare_features(
    df_real=val_data,
    fit=False,  # do not fit, just transform
    encoders=encoders,
    scaler=scaler,
    power_transformer=power_transformer
)

print("Validation features prepared:", val_features.shape)
