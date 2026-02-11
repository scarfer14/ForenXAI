"""
Machine Learning Models for Forensic Analysis
- Supervised: Random Forest, XGBoost, LightGBM
- Unsupervised: Isolation Forest, ECOD (Empirical CDF Outlier Detection)
Optimized for fast training on tabular network traffic data

CRITICAL FIX: Handles infinity and NaN values in data
GOOGLE COLAB OPTIMIZED: Won't crash on free tier (12GB RAM)

WHY ECOD instead of One-Class SVM?
- ECOD is parameter-free (no tuning needed)
- 100x faster on high-dimensional data
- More interpretable (based on empirical CDF)
- Better suited for network anomaly detection
- Proven effective for cybersecurity datasets
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import xgboost as xgb
import lightgbm as lgb
import time
import warnings
import gc  # CRITICAL: Garbage collection for memory management

# Professional warning handling: Suppress only specific known warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='X does not have valid feature names', category=UserWarning)
warnings.filterwarnings('ignore', message='.*LightGBM.*', category=UserWarning)

print("="*70)
print("MACHINE LEARNING MODEL TRAINING - PRODUCTION READY")
print("="*70)

# ============================================================
# DATA CLEANING FUNCTIONS
# ============================================================

def clean_data(X, name="data"):
    """
    Clean data by handling infinity and NaN values

    Args:
        X: numpy array or pandas DataFrame
        name: name for logging

    Returns:
        Cleaned numpy array
    """
    print(f"\n🧹 Cleaning {name}...")

    # Convert to numpy if DataFrame
    if isinstance(X, pd.DataFrame):
        X_clean = X.values.copy()
    else:
        X_clean = X.copy()

    # Check for issues
    n_inf = np.isinf(X_clean).sum()
    n_nan = np.isnan(X_clean).sum()

    if n_inf > 0 or n_nan > 0:
        print(f"   ⚠️  Found issues:")
        print(f"      Infinity values: {n_inf:,}")
        print(f"      NaN values: {n_nan:,}")

        # Replace infinity with large but finite values
        X_clean[np.isposinf(X_clean)] = np.finfo(np.float32).max / 2
        X_clean[np.isneginf(X_clean)] = np.finfo(np.float32).min / 2

        # Replace NaN with 0 (common for network features with no data)
        X_clean[np.isnan(X_clean)] = 0

        print(f"   ✅ Cleaned:")
        print(f"      Replaced +inf with {np.finfo(np.float32).max / 2:.2e}")
        print(f"      Replaced -inf with {np.finfo(np.float32).min / 2:.2e}")
        print(f"      Replaced NaN with 0")
    else:
        print(f"   ✅ No issues found (clean data)")

    # Final validation
    assert not np.isinf(X_clean).any(), "Still has infinity values!"
    assert not np.isnan(X_clean).any(), "Still has NaN values!"

    return X_clean

# ============================================================
# DATA LOADING WITH PROTECTION
# ============================================================
print("\n" + "="*70)
print("DATA LOADING & PROTECTION")
print("="*70)

# IMPORTANT: Set this to True if you want to force fresh data load
# Useful after runtime restart or if you suspect data corruption
FORCE_RELOAD = False  # Set to True to reload from CSV files

# Check if variables already exist from previous cells
required_vars = [
    'X_train_class', 'y_train_class',
    'X_val_class', 'y_val_class',
    'X_test_class', 'y_test_class',
    'X_train_anomaly', 'X_test_anomaly',
    'y_test_anomaly', 'train_trees_df'
]

variables_exist = all(var_name in globals() for var_name in required_vars)

if variables_exist and not FORCE_RELOAD:
    print("✅ Using existing data from previous cells")
    print("💡 TIP: Set FORCE_RELOAD=True above to load fresh data from CSV files")

    # Even with existing data, create protected copies AND CLEAN
    print("🛡️  Creating protected copies and cleaning data...")
    X_train_class = X_train_class.copy() if hasattr(X_train_class, 'copy') else np.array(X_train_class)
    y_train_class = y_train_class.copy() if hasattr(y_train_class, 'copy') else np.array(y_train_class)
    X_val_class = X_val_class.copy() if hasattr(X_val_class, 'copy') else np.array(X_val_class)
    y_val_class = y_val_class.copy() if hasattr(y_val_class, 'copy') else np.array(y_val_class)
    X_test_class = X_test_class.copy() if hasattr(X_test_class, 'copy') else np.array(X_test_class)
    y_test_class = y_test_class.copy() if hasattr(y_test_class, 'copy') else np.array(y_test_class)
    X_train_anomaly = X_train_anomaly.copy() if hasattr(X_train_anomaly, 'copy') else np.array(X_train_anomaly)
    X_test_anomaly = X_test_anomaly.copy() if hasattr(X_test_anomaly, 'copy') else np.array(X_test_anomaly)
    y_test_anomaly = y_test_anomaly.copy() if hasattr(y_test_anomaly, 'copy') else np.array(y_test_anomaly)

    # CRITICAL: Clean all data arrays
    X_train_class = clean_data(X_train_class, "X_train_class")
    X_val_class = clean_data(X_val_class, "X_val_class")
    X_test_class = clean_data(X_test_class, "X_test_class")
    X_train_anomaly = clean_data(X_train_anomaly, "X_train_anomaly")
    X_test_anomaly = clean_data(X_test_anomaly, "X_test_anomaly")

    print("✅ Protected copies created and cleaned")

else:
    if FORCE_RELOAD:
        print("🔄 FORCE_RELOAD enabled - loading fresh data from files...")
    else:
        print("⚠️  Variables not found - loading from processed files...")

    # Try to load from processed CSV files
    import os

    # Get script directory and navigate to data folder
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
    PROCESSED_DIR = os.path.join(ROOT_DIR, 'data', 'processed')

    train_path = os.path.join(PROCESSED_DIR, 'train_features.csv')
    val_path = os.path.join(PROCESSED_DIR, 'validation_features.csv')
    test_path = os.path.join(PROCESSED_DIR, 'test_features.csv')

    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        print("\n❌ ERROR: Processed data files not found!")
        print("\nYou need to run data preparation first:")
        print("  Option 1: Run data_prep.ipynb notebook cells")
        print("  Option 2: Run example_run_feature_prep.py script")
        print(f"\nExpected files in: {PROCESSED_DIR}")
        print("  - train_features.csv")
        print("  - validation_features.csv")
        print("  - test_features.csv")
        raise FileNotFoundError("Run data preparation before training models")

    # Load processed data with .copy() to protect original data
    print(f"\n📂 Loading from: {PROCESSED_DIR}")
    print("🛡️  Using .copy() for data protection (prevents in-place modifications)")
    train_df = pd.read_csv(train_path).copy()
    val_df = pd.read_csv(val_path).copy()
    test_df = pd.read_csv(test_path).copy()
    train_trees_df = train_df.copy()  # For feature column extraction

    print(f"✅ Loaded train: {train_df.shape}")
    print(f"✅ Loaded validation: {val_df.shape}")
    print(f"✅ Loaded test: {test_df.shape}")

    # Split features and labels (using .copy() to prevent reference issues)
    X_train_class = train_df.drop(columns=['label']).values.copy()
    y_train_class = train_df['label'].values.copy()

    X_val_class = val_df.drop(columns=['label']).values.copy()
    y_val_class = val_df['label'].values.copy()

    X_test_class = test_df.drop(columns=['label']).values.copy()
    y_test_class = test_df['label'].values.copy()

    # CRITICAL: Clean all data before use
    X_train_class = clean_data(X_train_class, "X_train_class")
    X_val_class = clean_data(X_val_class, "X_val_class")
    X_test_class = clean_data(X_test_class, "X_test_class")

    # For unsupervised training (anomaly detection)
    X_train_anomaly = X_train_class[y_train_class == 0].copy()
    X_test_anomaly = X_test_class.copy()
    y_test_anomaly = y_test_class.copy()

    # Clean anomaly detection data
    X_train_anomaly = clean_data(X_train_anomaly, "X_train_anomaly")
    X_test_anomaly = clean_data(X_test_anomaly, "X_test_anomaly")

    print("\n✅ Data prepared for training")
    print("🛡️  All arrays created with .copy() - original data protected")
    print("💡 Safe for multiple runtime restarts without data corruption")

# Verify all variables now exist
print("\n" + "="*70)
print("DATA VERIFICATION")
print("="*70)
print(f"\n✅ All data loaded and cleaned successfully")
print(f"   Training samples: {len(X_train_class):,}")
print(f"   Feature dimensions: {X_train_class.shape[1]}")
print(f"   Data type: {X_train_class.dtype}")
print(f"   Value range: [{X_train_class.min():.2e}, {X_train_class.max():.2e}]")

# Final validation check
assert not np.isinf(X_train_class).any(), "X_train_class has infinity!"
assert not np.isnan(X_train_class).any(), "X_train_class has NaN!"
assert not np.isinf(X_val_class).any(), "X_val_class has infinity!"
assert not np.isnan(X_val_class).any(), "X_val_class has NaN!"
assert not np.isinf(X_test_class).any(), "X_test_class has infinity!"
assert not np.isnan(X_test_class).any(), "X_test_class has NaN!"
print("✅ Data validation passed: No infinity or NaN values")

# ============================================================
# SETUP: Ensure we have feature column names
# ============================================================
# Get feature names from original dataframes
feature_cols = [col for col in train_trees_df.columns if col not in ['Label', 'Attack', 'label']]
print(f"\nNumber of features: {len(feature_cols)}")

# Convert NumPy arrays to DataFrames with column names (using .copy() for protection)
print("🛡️  Converting to DataFrames with data protection...")
X_train_class_df = pd.DataFrame(X_train_class.copy(), columns=feature_cols)
X_val_class_df = pd.DataFrame(X_val_class.copy(), columns=feature_cols)
X_test_class_df = pd.DataFrame(X_test_class.copy(), columns=feature_cols)

X_train_anomaly_df = pd.DataFrame(X_train_anomaly.copy(), columns=feature_cols)
X_test_anomaly_df = pd.DataFrame(X_test_anomaly.copy(), columns=feature_cols)

print("✅ All data converted to DataFrames with feature names (protected copies)")

# ============================================================
# PART 1: SUPERVISED CLASSIFICATION
# ============================================================
print("\n" + "="*70)
print("PART 1: SUPERVISED CLASSIFICATION (Normal vs Attack)")
print("="*70)

# Calculate class imbalance for boosting models
scale_pos_weight = np.sum(y_train_class == 0) / np.sum(y_train_class == 1)
print(f"\nClass imbalance ratio: {scale_pos_weight:.2f}:1")
print(f"Training on {len(y_train_class):,} samples")

supervised_results = {}

# ------------------------------------------------------------
# 1. Random Forest (Baseline)
# ------------------------------------------------------------
print("\n[1/3] Training Random Forest...")
start = time.time()

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

# Use DataFrame for training
rf_model.fit(X_train_class_df, y_train_class)
rf_train_time = time.time() - start

# Predictions
rf_test_pred = rf_model.predict(X_test_class_df)
rf_test_proba = rf_model.predict_proba(X_test_class_df)[:, 1]

# Metrics
supervised_results['Random Forest'] = {
    'Accuracy': accuracy_score(y_test_class, rf_test_pred),
    'Precision': precision_score(y_test_class, rf_test_pred),
    'Recall': recall_score(y_test_class, rf_test_pred),
    'F1-Score': f1_score(y_test_class, rf_test_pred),
    'ROC-AUC': roc_auc_score(y_test_class, rf_test_proba),
    'Training Time': rf_train_time
}

print(f"✅ Trained in {rf_train_time:.2f}s")
print(f"   F1-Score: {supervised_results['Random Forest']['F1-Score']:.4f}")
print(f"   ROC-AUC: {supervised_results['Random Forest']['ROC-AUC']:.4f}")

# Free memory
gc.collect()

# ------------------------------------------------------------
# 2. XGBoost (Advanced Boosting)
# ------------------------------------------------------------
print("\n[2/3] Training XGBoost...")
start = time.time()

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
    verbosity=0
)

# Use DataFrame for training
xgb_model.fit(
    X_train_class_df,
    y_train_class,
    eval_set=[(X_val_class_df, y_val_class)],
    verbose=False
)
xgb_train_time = time.time() - start

# Predictions
xgb_test_pred = xgb_model.predict(X_test_class_df)
xgb_test_proba = xgb_model.predict_proba(X_test_class_df)[:, 1]

# Metrics
supervised_results['XGBoost'] = {
    'Accuracy': accuracy_score(y_test_class, xgb_test_pred),
    'Precision': precision_score(y_test_class, xgb_test_pred),
    'Recall': recall_score(y_test_class, xgb_test_pred),
    'F1-Score': f1_score(y_test_class, xgb_test_pred),
    'ROC-AUC': roc_auc_score(y_test_class, xgb_test_proba),
    'Training Time': xgb_train_time
}

print(f"✅ Trained in {xgb_train_time:.2f}s")
print(f"   F1-Score: {supervised_results['XGBoost']['F1-Score']:.4f}")
print(f"   ROC-AUC: {supervised_results['XGBoost']['ROC-AUC']:.4f}")

# Free memory
gc.collect()

# ------------------------------------------------------------
# 3. LightGBM (Fast Boosting)
# ------------------------------------------------------------
print("\n[3/3] Training LightGBM...")
start = time.time()

lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.1,
    class_weight='balanced',
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# Use DataFrame for training and prediction
lgb_model.fit(
    X_train_class_df,  # DataFrame with column names
    y_train_class,
    eval_set=[(X_val_class_df, y_val_class)],  # DataFrame for validation
    callbacks=[lgb.early_stopping(50, verbose=False)]
)
lgb_train_time = time.time() - start

# Use DataFrame for predictions
lgb_test_pred = lgb_model.predict(X_test_class_df)  # DataFrame
lgb_test_proba = lgb_model.predict_proba(X_test_class_df)[:, 1]  # DataFrame

# Metrics
supervised_results['LightGBM'] = {
    'Accuracy': accuracy_score(y_test_class, lgb_test_pred),
    'Precision': precision_score(y_test_class, lgb_test_pred),
    'Recall': recall_score(y_test_class, lgb_test_pred),
    'F1-Score': f1_score(y_test_class, lgb_test_pred),
    'ROC-AUC': roc_auc_score(y_test_class, lgb_test_proba),
    'Training Time': lgb_train_time
}

print(f"✅ Trained in {lgb_train_time:.2f}s")
print(f"   F1-Score: {supervised_results['LightGBM']['F1-Score']:.4f}")
print(f"   ROC-AUC: {supervised_results['LightGBM']['ROC-AUC']:.4f}")

# Free memory
gc.collect()

# ============================================================
# PART 2: UNSUPERVISED ANOMALY DETECTION
# ============================================================
print("\n" + "="*70)
print("PART 2: UNSUPERVISED ANOMALY DETECTION (Outlier Discovery)")
print("="*70)
print(f"Training on {len(X_train_anomaly):,} samples (no labels used)")

unsupervised_results = {}

# ------------------------------------------------------------
# 1. Isolation Forest (Tree-based Baseline)
# ------------------------------------------------------------
print("\n[1/2] Training Isolation Forest...")
start = time.time()

iso_model = IsolationForest(
    n_estimators=100,
    max_samples='auto',
    contamination=0.05,  # Expected 5% outliers
    random_state=42,
    n_jobs=-1,
    verbose=0
)

# NumPy array is fine for Isolation Forest (tree-based)
iso_model.fit(X_train_anomaly)
iso_train_time = time.time() - start

# Predictions (use test set with labels for evaluation)
iso_scores = iso_model.score_samples(X_test_anomaly)
iso_pred = iso_model.predict(X_test_anomaly)
iso_pred_binary = np.where(iso_pred == -1, 1, 0)

# Metrics
unsupervised_results['Isolation Forest'] = {
    'Accuracy': accuracy_score(y_test_anomaly, iso_pred_binary),
    'Precision': precision_score(y_test_anomaly, iso_pred_binary, zero_division=0),
    'Recall': recall_score(y_test_anomaly, iso_pred_binary),
    'F1-Score': f1_score(y_test_anomaly, iso_pred_binary),
    'ROC-AUC': roc_auc_score(y_test_anomaly, -iso_scores),
    'Training Time': iso_train_time
}

print(f"✅ Trained in {iso_train_time:.2f}s")
print(f"   F1-Score: {unsupervised_results['Isolation Forest']['F1-Score']:.4f}")
print(f"   ROC-AUC: {unsupervised_results['Isolation Forest']['ROC-AUC']:.4f}")

# Free memory after Isolation Forest
gc.collect()

# ------------------------------------------------------------
# 2. ECOD (Empirical Cumulative Distribution Outlier Detection)
# ------------------------------------------------------------
print("\n[2/2] Training ECOD (Empirical CDF Outlier Detection)...")
print("   🎯 Google Colab Optimized - Won't crash on free tier")

try:
    from pyod.models.ecod import ECOD

    # ========================================================
    # CRITICAL: Memory-safe configuration for Google Colab
    # ========================================================

    # Step 1: Analyze dataset dimensions
    n_features = X_train_anomaly.shape[1]
    n_samples = len(X_train_anomaly)

    print(f"   Dataset: {n_samples:,} samples × {n_features} features")

    # Step 2: Set memory-safe limits based on empirical testing
    # Colab free tier has ~12GB RAM - these limits are tested to work
    if n_features > 100:
        max_ecod_samples = 40000   # High-dimensional: Conservative limit
        batch_size = 30000
    elif n_features > 50:
        max_ecod_samples = 80000   # Medium-dimensional
        batch_size = 40000
    else:
        max_ecod_samples = 120000  # Low-dimensional
        batch_size = 50000

    print(f"   Memory limits: max_train={max_ecod_samples:,}, batch_size={batch_size:,}")

    # Step 3: Sample training data if needed (CRITICAL: use .copy())
    if n_samples > max_ecod_samples:
        print(f"   📉 Sampling {max_ecod_samples:,} from {n_samples:,} samples (prevents OOM)")
        np.random.seed(42)
        sample_idx = np.random.choice(n_samples, max_ecod_samples, replace=False)
        X_train_ecod = X_train_anomaly[sample_idx].copy()  # CRITICAL: .copy()
    else:
        print(f"   ✅ Using full dataset: {n_samples:,} samples")
        X_train_ecod = X_train_anomaly.copy()  # CRITICAL: .copy()

    # Step 4: Force garbage collection before training
    gc.collect()

    # Step 5: Train ECOD with Colab-stable settings
    print(f"   🔧 Training ECOD model...")
    start = time.time()

    ecod_model = ECOD(
        contamination=0.05,
        n_jobs=1  # CRITICAL: Single-threaded for Colab stability
    )

    ecod_model.fit(X_train_ecod)
    ecod_train_time = time.time() - start

    print(f"   ✅ Model trained in {ecod_train_time:.2f}s")

    # Step 6: Batch predictions to prevent memory spikes
    n_test = len(X_test_anomaly)

    if n_test > batch_size:
        print(f"   📦 Batch prediction: {n_test:,} samples in chunks of {batch_size:,}")

        ecod_scores = []
        ecod_pred = []

        n_batches = (n_test + batch_size - 1) // batch_size  # Ceiling division

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_test)

            if batch_idx % 5 == 0 or batch_idx == n_batches - 1:  # Progress updates
                print(f"      Batch {batch_idx+1}/{n_batches}: Processing {end_idx:,}/{n_test:,} samples")

            batch = X_test_anomaly[start_idx:end_idx]

            # Predict on batch
            scores_batch = ecod_model.decision_function(batch)
            pred_batch = ecod_model.predict(batch)

            ecod_scores.extend(scores_batch)
            ecod_pred.extend(pred_batch)

            # Free memory between batches
            del batch, scores_batch, pred_batch
            gc.collect()

        ecod_scores = np.array(ecod_scores)
        ecod_pred = np.array(ecod_pred)
        print(f"   ✅ Batch prediction complete")

    else:
        # Small test set - predict all at once
        print(f"   ✅ Single-pass prediction: {n_test:,} samples")
        ecod_scores = ecod_model.decision_function(X_test_anomaly)
        ecod_pred = ecod_model.predict(X_test_anomaly)

    # Step 7: Calculate metrics
    unsupervised_results['ECOD'] = {
        'Accuracy': accuracy_score(y_test_anomaly, ecod_pred),
        'Precision': precision_score(y_test_anomaly, ecod_pred, zero_division=0),
        'Recall': recall_score(y_test_anomaly, ecod_pred),
        'F1-Score': f1_score(y_test_anomaly, ecod_pred),
        'ROC-AUC': roc_auc_score(y_test_anomaly, ecod_scores),
        'Training Time': ecod_train_time
    }

    print(f"   ✅ ECOD Results:")
    print(f"      F1-Score: {unsupervised_results['ECOD']['F1-Score']:.4f}")
    print(f"      ROC-AUC: {unsupervised_results['ECOD']['ROC-AUC']:.4f}")
    print(f"      Training Time: {ecod_train_time:.2f}s")

    # Speed comparison
    if 'Isolation Forest' in unsupervised_results:
        speed_ratio = iso_train_time / ecod_train_time if ecod_train_time > 0 else 1
        if speed_ratio > 1:
            print(f"      Speed: {speed_ratio:.1f}x faster than Isolation Forest")
        elif speed_ratio < 1:
            print(f"      Speed: {1/speed_ratio:.1f}x slower than Isolation Forest")
        else:
            print(f"      Speed: Similar to Isolation Forest")

    print(f"   📊 Summary:")
    print(f"      Trained on: {len(X_train_ecod):,} samples")
    print(f"      Evaluated on: {len(X_test_anomaly):,} test samples")
    print(f"      Memory-safe: ✅ Colab-optimized")

    # Step 8: Critical cleanup
    del X_train_ecod, ecod_scores, ecod_pred
    gc.collect()

    ecod_available = True
    print(f"   ✅ ECOD completed successfully (no crashes)")

except ImportError:
    print("\n   ⚠️  PyOD library not installed")
    print("      Install: pip install pyod")
    print("      Skipping ECOD... Using Isolation Forest only")
    ecod_available = False

except MemoryError as e:
    print(f"\n   ❌ ECOD failed: Out of memory")
    print(f"      Dataset: {len(X_train_anomaly):,} samples × {X_train_anomaly.shape[1]} features")
    print(f"      This shouldn't happen with Colab-optimized limits")
    print(f"      Try: Restart runtime and reduce max_ecod_samples to 30,000")
    ecod_available = False

    # Critical cleanup
    if 'X_train_ecod' in locals():
        del X_train_ecod
    gc.collect()

except Exception as e:
    print(f"\n   ❌ ECOD failed: {type(e).__name__}")
    print(f"      Error: {str(e)[:200]}")  # Truncate long errors
    print(f"      Possible causes:")
    print(f"         - Data type incompatibility")
    print(f"         - PyOD version mismatch (try: pip install --upgrade pyod)")
    print(f"         - Corrupted data arrays")
    print(f"      Skipping ECOD... Using Isolation Forest only")
    ecod_available = False

    # Critical cleanup
    if 'X_train_ecod' in locals():
        del X_train_ecod
    gc.collect()

# Final garbage collection
gc.collect()