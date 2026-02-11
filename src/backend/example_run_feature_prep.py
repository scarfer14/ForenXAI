import pandas as pd
import joblib
import os
from feature_prep import prepare_features

# Navigate to ForenXAI root directory (2 levels up from src/backend)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, '..', '..')
ROOT_DIR = os.path.abspath(ROOT_DIR)  # Resolve .. properly
os.chdir(ROOT_DIR)

print(f"Working directory: {os.getcwd()}")
print(f"Checking if data/real exists: {os.path.exists('data/real')}")
print(f"Output will be saved to:")
print(f"  - {os.path.abspath('src/backend/feature_engineering')}")
print(f"  - {os.path.abspath('data/processed')}")
print()

# ===============================
# Load datasets from data/real/ and data/synthetic/
# ===============================
TRAIN_PATH = 'data/real/train_real.csv'
VAL_PATH = 'data/real/val_real.csv'
TEST_PATH = 'data/real/test_real.csv'

# Load synthetic data
SYNTHETIC_TRAIN_PATH = 'data/synthetic/synthetic_train_split.csv'

print("=" * 70)
print("LOADING CLEANED DATA")
print("=" * 70)

# Load training data with memory optimization
print("Loading train data (this may take a moment)...")
train_real = pd.read_csv(TRAIN_PATH, low_memory=False)
print(f"✅ Train data loaded: {train_real.shape}")

# Load synthetic data if available
train_synthetic = None
if SYNTHETIC_TRAIN_PATH and os.path.exists(SYNTHETIC_TRAIN_PATH):
    train_synthetic = pd.read_csv(SYNTHETIC_TRAIN_PATH, low_memory=False)
    print(f"✅ Synthetic train data loaded: {train_synthetic.shape}")
else:
    print("⚠️  No synthetic data found (optional)")

# Load validation data
val_data = pd.read_csv(VAL_PATH, low_memory=False)
print(f"✅ Validation data loaded: {val_data.shape}")

# Load test data
test_data = pd.read_csv(TEST_PATH, low_memory=False)
print(f"✅ Test data loaded: {test_data.shape}")

print("\n" + "=" * 70)
print("COLUMNS IN DATA:")
print("=" * 70)
print(train_real.columns.tolist())

# ===============================
# TRAINING PIPELINE
# ===============================
print("\n" + "=" * 70)
print("FEATURE ENGINEERING (This may take 5-15 minutes...)")
print("=" * 70)
print("⏳ Step 1/5: Combining datasets...")

train_outputs, artifacts = prepare_features(
    df_real=train_real,
    df_synthetic=train_synthetic,
    fit=True
)

train_trees = train_outputs['tree_models']   # For RF / Isolation Forest
train_dl = train_outputs['deep_learning']    # For Deep Learning

print("Train (trees):", train_trees.shape)
print("Train (DL):", train_dl.shape)

# ===============================
# VALIDATION PIPELINE
# ===============================
print("\n" + "=" * 70)
print("PROCESSING VALIDATION DATA")
print("=" * 70)

# Load artifacts using absolute paths
FEATURE_ENG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feature_engineering')
encoders = joblib.load(os.path.join(FEATURE_ENG_DIR, 'encoders/label_encoders.joblib'))
scaler = joblib.load(os.path.join(FEATURE_ENG_DIR, 'scalers/scaler.joblib'))
power_transformer = joblib.load(os.path.join(FEATURE_ENG_DIR, 'scalers/power_transformer.joblib'))

val_outputs, _ = prepare_features(
    df_real=val_data,
    fit=False,
    encoders=encoders,
    scaler=scaler,
    power_transformer=power_transformer
)

val_trees = val_outputs['tree_models']
val_dl = val_outputs['deep_learning']

print("Val (trees):", val_trees.shape)
print("Val (DL):", val_dl.shape)

# ===============================
# TEST PIPELINE
# ===============================
print("\n" + "=" * 70)
print("PROCESSING TEST DATA")
print("=" * 70)

test_outputs, _ = prepare_features(
    df_real=test_data,
    fit=False,
    encoders=encoders,
    scaler=scaler,
    power_transformer=power_transformer
)

test_trees = test_outputs['tree_models']
test_dl = test_outputs['deep_learning']

print("Test (trees):", test_trees.shape)
print("Test (DL):", test_dl.shape)

# ===============================
# SAVE ENGINEERED FEATURES
# ===============================
print("\n" + "=" * 70)
print("SAVING ENGINEERED FEATURES")
print("=" * 70)

# Create processed folder if it doesn't exist
PROCESSED_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'data', 'processed'))
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Save BOTH versions: unscaled (for tree models) and scaled (for deep learning)
# Tree models (RF, XGBoost, Isolation Forest) - UNSCALED
train_trees.to_csv(os.path.join(PROCESSED_DIR, 'train_features.csv'), index=False)
val_trees.to_csv(os.path.join(PROCESSED_DIR, 'validation_features.csv'), index=False)
test_trees.to_csv(os.path.join(PROCESSED_DIR, 'test_features.csv'), index=False)

# Deep learning models (MLPs, VAEs) - SCALED
train_dl.to_csv(os.path.join(PROCESSED_DIR, 'train_features_dl.csv'), index=False)
val_dl.to_csv(os.path.join(PROCESSED_DIR, 'validation_features_dl.csv'), index=False)
test_dl.to_csv(os.path.join(PROCESSED_DIR, 'test_features_dl.csv'), index=False)

print(f"✅ Saved (UNSCALED for tree models):")
print(f"   {os.path.join(PROCESSED_DIR, 'train_features.csv')}")
print(f"   {os.path.join(PROCESSED_DIR, 'validation_features.csv')}")
print(f"   {os.path.join(PROCESSED_DIR, 'test_features.csv')}")
print(f"\n✅ Saved (SCALED for deep learning):")
print(f"   {os.path.join(PROCESSED_DIR, 'train_features_dl.csv')}")
print(f"   {os.path.join(PROCESSED_DIR, 'validation_features_dl.csv')}")
print(f"   {os.path.join(PROCESSED_DIR, 'test_features_dl.csv')}")
print("\n🎉 Feature engineering complete!")
print("   - Use train_features.csv for tree models (RF, XGBoost, Isolation Forest)")
print("   - Use train_features_dl.csv for deep learning models (MLPs, VAEs)")
