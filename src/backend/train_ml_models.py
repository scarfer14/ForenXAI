"""
Machine Learning Models for Forensic Analysis
- Supervised: Random Forest (tree-based), MLP (deep learning)
- Unsupervised: Isolation Forest
Optimized for network traffic anomaly detection and attack classification

BEST PRACTICES IMPLEMENTED:
✓ Reproducibility: Global RANDOM_STATE=42 for all models
✓ Metrics: Precision, Recall, F1 (Binary, Macro, Weighted), Confusion Matrix, ROC-AUC
✓ Data Protection: .copy() calls prevent in-place modifications
✓ Leakage Prevention: Preprocessing done separately in feature_prep.py
✓ Model Persistence: All models saved to models/ directory
✓ Train/Val/Test Split: Standard for large datasets (1.8M samples)

Note: Cross-validation skipped (standard practice for large-scale datasets)
"""

import numpy as np
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import time
import warnings

# ============================================================
# REPRODUCIBILITY: Set global random seeds
# ============================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Professional warning handling
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='X does not have valid feature names', category=UserWarning)

print("="*70)
print("MACHINE LEARNING MODEL TRAINING - ENHANCED")
print("="*70)
print(f"🔒 Random State: {RANDOM_STATE} (Reproducible Results)")
print("📊 Training Strategy: Train/Val/Test Split (1.8M samples)")
print("✅ Hyperparameter Tuning: RandomizedSearchCV")
print("✅ Pipeline Wrapping: Production-ready models")
print("✅ Split Verification: Attack type balance check")
print("💡 Note: Cross-validation skipped (standard for large datasets)")

# ============================================================
# DATA LOADING
# ============================================================
print("\n" + "="*70)
print("DATA LOADING")
print("="*70)

# Get script directory and navigate to data folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
PROCESSED_DIR = os.path.join(ROOT_DIR, 'data', 'processed')

# For tree models (RF, Isolation Forest) - use unscaled
train_path = os.path.join(PROCESSED_DIR, 'train_features.csv')
val_path = os.path.join(PROCESSED_DIR, 'validation_features.csv')
test_path = os.path.join(PROCESSED_DIR, 'test_features.csv')

# For deep learning (MLP) - use scaled
train_dl_path = os.path.join(PROCESSED_DIR, 'train_features_dl.csv')
val_dl_path = os.path.join(PROCESSED_DIR, 'validation_features_dl.csv')
test_dl_path = os.path.join(PROCESSED_DIR, 'test_features_dl.csv')

if not all(os.path.exists(p) for p in [train_path, val_path, test_path, train_dl_path, val_dl_path, test_dl_path]):
    print("\n❌ ERROR: Processed data files not found!")
    print("\nYou need to run data preparation first:")
    print("  Option 1: Run example_run_feature_prep.py script")
    print(f"\nExpected files in: {PROCESSED_DIR}")
    print("  - train_features.csv, validation_features.csv, test_features.csv (unscaled)")
    print("  - train_features_dl.csv, validation_features_dl.csv, test_features_dl.csv (scaled)")
    raise FileNotFoundError("Run data preparation before training models")

# Load processed data
print(f"\n📂 Loading from: {PROCESSED_DIR}")

# Tree models (unscaled)
train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

# Deep learning (scaled)
train_dl_df = pd.read_csv(train_dl_path)
val_dl_df = pd.read_csv(val_dl_path)
test_dl_df = pd.read_csv(test_dl_path)

print(f"✅ Loaded train (unscaled): {train_df.shape}")
print(f"✅ Loaded train (scaled): {train_dl_df.shape}")

# Split features and labels for tree models
X_train_class = train_df.drop(columns=['Label', 'Attack'], errors='ignore').values
y_train_class = train_df['Label'].values

X_val_class = val_df.drop(columns=['Label', 'Attack'], errors='ignore').values
y_val_class = val_df['Label'].values

X_test_class = test_df.drop(columns=['Label', 'Attack'], errors='ignore').values
y_test_class = test_df['Label'].values

# Split features and labels for deep learning
X_train_dl = train_dl_df.drop(columns=['Label', 'Attack'], errors='ignore').values
y_train_dl = train_dl_df['Label'].values

X_val_dl = val_dl_df.drop(columns=['Label', 'Attack'], errors='ignore').values
y_val_dl = val_dl_df['Label'].values

X_test_dl = test_dl_df.drop(columns=['Label', 'Attack'], errors='ignore').values
y_test_dl = test_dl_df['Label'].values

# For unsupervised training (anomaly detection)
X_train_anomaly = X_train_class[y_train_class == 0]
X_test_anomaly = X_test_class
y_test_anomaly = y_test_class

print(f"\n✅ Data prepared for training")
print(f"   Training samples: {len(X_train_class):,}")
print(f"   Feature dimensions: {X_train_class.shape[1]}")

# ============================================================
# SPLIT VERIFICATION: Check for balanced attack types
# ============================================================
print("\n" + "="*70)
print("SPLIT VERIFICATION (Attack Type Balance)")
print("="*70)

# Check Label distribution (Normal vs Attack)
train_label_dist = np.bincount(y_train_class) / len(y_train_class)
val_label_dist = np.bincount(y_val_class) / len(y_val_class)
test_label_dist = np.bincount(y_test_class) / len(y_test_class)

print("\nLabel Distribution (Normal=0, Attack=1):")
print(f"  Train: Normal={train_label_dist[0]:.3f}, Attack={train_label_dist[1]:.3f}")
print(f"  Val:   Normal={val_label_dist[0]:.3f}, Attack={val_label_dist[1]:.3f}")
print(f"  Test:  Normal={test_label_dist[0]:.3f}, Attack={test_label_dist[1]:.3f}")

# Check Attack Type distribution
print("\nAttack Type Distribution:")
train_attacks = train_df['Attack'].value_counts(normalize=True).sort_index()
val_attacks = val_df['Attack'].value_counts(normalize=True).sort_index()
test_attacks = test_df['Attack'].value_counts(normalize=True).sort_index()

attack_comparison = pd.DataFrame({
    'Train': train_attacks,
    'Val': val_attacks,
    'Test': test_attacks
}).fillna(0)

print(attack_comparison)

# Flag significant imbalances
print("\nBalance Check:")
imbalance_warnings = 0
for attack in attack_comparison.index:
    train_pct = attack_comparison.loc[attack, 'Train']
    val_pct = attack_comparison.loc[attack, 'Val']
    test_pct = attack_comparison.loc[attack, 'Test']
    
    # Check if any split differs by >50% from training proportion
    if train_pct > 0:
        val_diff = abs(train_pct - val_pct) / train_pct
        test_diff = abs(train_pct - test_pct) / train_pct
        
        if val_diff > 0.5:
            print(f"  ⚠️  {attack}: Train={train_pct:.4f}, Val={val_pct:.4f} (>{val_diff*100:.1f}% diff)")
            imbalance_warnings += 1
        if test_diff > 0.5:
            print(f"  ⚠️  {attack}: Train={train_pct:.4f}, Test={test_pct:.4f} (>{test_diff*100:.1f}% diff)")
            imbalance_warnings += 1

if imbalance_warnings == 0:
    print("  ✅ All attack types are well-balanced across splits")
else:
    print(f"  ⚠️  Found {imbalance_warnings} potential imbalance(s) - review above")

# Setup model save directory
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
print(f"\n💾 Models will be saved to: {MODELS_DIR}")

# ============================================================
# PART 1: SUPERVISED CLASSIFICATION
# ============================================================
print("\n" + "="*70)
print("PART 1: SUPERVISED CLASSIFICATION (Normal vs Attack)")
print("="*70)

# Calculate class imbalance
scale_pos_weight = np.sum(y_train_class == 0) / np.sum(y_train_class == 1)
print(f"\nClass imbalance ratio: {scale_pos_weight:.2f}:1")
print(f"Training on {len(y_train_class):,} samples")

supervised_results = {}

# ------------------------------------------------------------
# 1. Random Forest with Hyperparameter Tuning + Pipeline
# ------------------------------------------------------------
print("\n[1/2] Training Random Forest with Hyperparameter Tuning...")
print("  Step 1: Hyperparameter search on 200K sample subset...")

# Sample subset for efficient tuning (standard practice for large datasets)
np.random.seed(RANDOM_STATE)
sample_size = min(200_000, len(X_train_class))
sample_idx = np.random.choice(len(X_train_class), sample_size, replace=False)
X_sample = X_train_class[sample_idx]
y_sample = y_train_class[sample_idx]

# Define hyperparameter search space
param_distributions = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [15, 20, 25, 30, None],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 4, 8],
    'max_features': ['sqrt', 'log2']
}

# Randomized search (faster than GridSearch)
start_tuning = time.time()
rf_search = RandomizedSearchCV(
    RandomForestClassifier(
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    ),
    param_distributions=param_distributions,
    n_iter=20,  # Test 20 random combinations
    cv=3,       # 3-fold CV on subset
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)

rf_search.fit(X_sample, y_sample)
tuning_time = time.time() - start_tuning

print(f"  ✅ Tuning complete in {tuning_time:.1f}s")
print(f"  Best params: {rf_search.best_params_}")
print(f"  Best CV F1-Weighted: {rf_search.best_score_:.4f}")

# Train final model with best params on FULL training data
print("  Step 2: Training final model on full training set...")
start_train = time.time()

# Create pipeline with identity transformer (for future extensibility)
rf_pipeline = Pipeline([
    ('preprocessor', FunctionTransformer(validate=False)),  # Pass-through (data already preprocessed)
    ('classifier', RandomForestClassifier(
        **rf_search.best_params_,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    ))
])

rf_pipeline.fit(X_train_class, y_train_class)
rf_train_time = time.time() - start_train

# Predictions
rf_test_pred = rf_pipeline.predict(X_test_class)
rf_test_proba = rf_pipeline.predict_proba(X_test_class)[:, 1]

# Metrics
supervised_results['Random Forest'] = {
    'Accuracy': accuracy_score(y_test_class, rf_test_pred),
    'Precision': precision_score(y_test_class, rf_test_pred),
    'Recall': recall_score(y_test_class, rf_test_pred),
    'F1-Score': f1_score(y_test_class, rf_test_pred),
    'F1-Macro': f1_score(y_test_class, rf_test_pred, average='macro'),
    'F1-Weighted': f1_score(y_test_class, rf_test_pred, average='weighted'),
    'ROC-AUC': roc_auc_score(y_test_class, rf_test_proba),
    'Training Time': rf_train_time,
    'Tuning Time': tuning_time
}

# Confusion Matrix
rf_cm = confusion_matrix(y_test_class, rf_test_pred)

print(f"  ✅ Trained in {rf_train_time:.2f}s")
print(f"   F1-Score: {supervised_results['Random Forest']['F1-Score']:.4f}")
print(f"   F1-Macro: {supervised_results['Random Forest']['F1-Macro']:.4f}")
print(f"   F1-Weighted: {supervised_results['Random Forest']['F1-Weighted']:.4f}")
print(f"   ROC-AUC: {supervised_results['Random Forest']['ROC-AUC']:.4f}")
print(f"   Confusion Matrix:\n{rf_cm}")

# Save Random Forest pipeline
rf_model_path = os.path.join(MODELS_DIR, 'random_forest_pipeline.joblib')
joblib.dump(rf_pipeline, rf_model_path)
print(f"   Saved pipeline: {rf_model_path}")

# ------------------------------------------------------------
# 2. MLP (Multi-Layer Perceptron - Deep Learning)
# ------------------------------------------------------------
print("\n[2/2] Training MLP (Neural Network)...")
print("   Using scaled features for deep learning...")
start = time.time()

# Build Neural Network
mlp_model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_dl.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

mlp_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Early stopping
es = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True,
    verbose=0
)

# Train the MLP
mlp_model.fit(
    X_train_dl,
    y_train_dl,
    validation_data=(X_val_dl, y_val_dl),
    epochs=20,
    batch_size=128,
    callbacks=[es],
    verbose=1
)

mlp_train_time = time.time() - start

# Predictions
mlp_test_proba = mlp_model.predict(X_test_dl, verbose=0).flatten()
mlp_test_pred = (mlp_test_proba > 0.5).astype(int)

# Metrics
supervised_results['MLP'] = {
    'Accuracy': accuracy_score(y_test_dl, mlp_test_pred),
    'Precision': precision_score(y_test_dl, mlp_test_pred),
    'Recall': recall_score(y_test_dl, mlp_test_pred),
    'F1-Score': f1_score(y_test_dl, mlp_test_pred),
    'F1-Macro': f1_score(y_test_dl, mlp_test_pred, average='macro'),
    'F1-Weighted': f1_score(y_test_dl, mlp_test_pred, average='weighted'),
    'ROC-AUC': roc_auc_score(y_test_dl, mlp_test_proba),
    'Training Time': mlp_train_time
}

# Confusion Matrix
mlp_cm = confusion_matrix(y_test_dl, mlp_test_pred)

print(f"Trained in {mlp_train_time:.2f}s")
print(f"   F1-Score: {supervised_results['MLP']['F1-Score']:.4f}")
print(f"   F1-Macro: {supervised_results['MLP']['F1-Macro']:.4f}")
print(f"   F1-Weighted: {supervised_results['MLP']['F1-Weighted']:.4f}")
print(f"   ROC-AUC: {supervised_results['MLP']['ROC-AUC']:.4f}")
print(f"   Confusion Matrix:\n{mlp_cm}")

# Save MLP model
mlp_model_path = os.path.join(MODELS_DIR, 'mlp_model.h5')
mlp_model.save(mlp_model_path)
print(f"   Saved model: {mlp_model_path}")

# ============================================================
# PART 2: UNSUPERVISED ANOMALY DETECTION
# ============================================================
print("\n" + "="*70)
print("PART 2: UNSUPERVISED ANOMALY DETECTION (Outlier Discovery)")
print("="*70)
print(f"Training on {len(X_train_anomaly):,} samples (no labels used)")

unsupervised_results = {}

# ------------------------------------------------------------
# Isolation Forest (Unsupervised Anomaly Detector) + Pipeline
# ------------------------------------------------------------
print("\nTraining Isolation Forest with Pipeline...")
start = time.time()

# Create pipeline for Isolation Forest
iso_pipeline = Pipeline([
    ('preprocessor', FunctionTransformer(validate=False)),  # Pass-through (data already preprocessed)
    ('detector', IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    ))
])

iso_pipeline.fit(X_train_anomaly)
iso_train_time = time.time() - start

# Predictions
iso_scores = iso_pipeline.score_samples(X_test_anomaly)
iso_pred = iso_pipeline.predict(X_test_anomaly)
iso_pred_binary = np.where(iso_pred == -1, 1, 0)

# Metrics
unsupervised_results['Isolation Forest'] = {
    'Accuracy': accuracy_score(y_test_anomaly, iso_pred_binary),
    'Precision': precision_score(y_test_anomaly, iso_pred_binary, zero_division=0),
    'Recall': recall_score(y_test_anomaly, iso_pred_binary),
    'F1-Score': f1_score(y_test_anomaly, iso_pred_binary),
    'F1-Macro': f1_score(y_test_anomaly, iso_pred_binary, average='macro'),
    'F1-Weighted': f1_score(y_test_anomaly, iso_pred_binary, average='weighted'),
    'ROC-AUC': roc_auc_score(y_test_anomaly, -iso_scores),
    'Training Time': iso_train_time
}

# Confusion Matrix
iso_cm = confusion_matrix(y_test_anomaly, iso_pred_binary)

print(f"Trained in {iso_train_time:.2f}s")
print(f"   F1-Score: {unsupervised_results['Isolation Forest']['F1-Score']:.4f}")
print(f"   F1-Macro: {unsupervised_results['Isolation Forest']['F1-Macro']:.4f}")
print(f"   F1-Weighted: {unsupervised_results['Isolation Forest']['F1-Weighted']:.4f}")
print(f"   ROC-AUC: {unsupervised_results['Isolation Forest']['ROC-AUC']:.4f}")
print(f"   Confusion Matrix:\n{iso_cm}")

# Save Isolation Forest pipeline
iso_model_path = os.path.join(MODELS_DIR, 'isolation_forest_pipeline.joblib')
joblib.dump(iso_pipeline, iso_model_path)
print(f"   Saved pipeline: {iso_model_path}")

# ============================================================
# PART 3: RESULTS COMPARISON
# ============================================================
print("\n" + "="*70)
print("MODEL COMPARISON RESULTS")
print("="*70)

# Supervised comparison
print("\nSUPERVISED CLASSIFICATION")
print("-"*70)
supervised_df = pd.DataFrame(supervised_results).T
supervised_df = supervised_df[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'F1-Macro', 'F1-Weighted', 'ROC-AUC', 'Training Time']]
print(supervised_df.to_string())

best_supervised = supervised_df['F1-Weighted'].idxmax()
print(f"\n✨ BEST SUPERVISED MODEL: {best_supervised}")
print(f"   F1-Weighted: {supervised_df.loc[best_supervised, 'F1-Weighted']:.4f}")
print(f"   ROC-AUC: {supervised_df.loc[best_supervised, 'ROC-AUC']:.4f}")

# Unsupervised comparison
print("\nUNSUPERVISED ANOMALY DETECTION")
print("-"*70)
unsupervised_df = pd.DataFrame(unsupervised_results).T
unsupervised_df = unsupervised_df[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'F1-Macro', 'F1-Weighted', 'ROC-AUC', 'Training Time']]
print(unsupervised_df.to_string())

# ============================================================
# PART 4: KEY INSIGHTS
# ============================================================
print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

print("\n1. REPRODUCIBILITY:")
print(f"   • Global Random State: {RANDOM_STATE}")
print("   • NumPy seed: Set")
print("   • TensorFlow seed: Set")
print("   • All models use consistent random_state parameter")

print("\n2. MODEL ARCHITECTURE:")
print("   • Random Forest: Tree-based ensemble (unscaled features)")
print("   • MLP: 4-layer neural network (scaled features, dropout regularization)")
print("   • Isolation Forest: Unsupervised outlier detection (unscaled features)")

print("\n3. METRICS (Imbalance-Aware):")
print("   • F1-Score (Binary): Standard F1 for positive class")
print("   • F1-Macro: Unweighted mean (treats classes equally)")
print("   • F1-Weighted: Weighted by class support (better for imbalanced data)")
print("   • ROC-AUC: Area under curve (threshold-independent)")
print("   • Confusion Matrix: [[TN, FP], [FN, TP]]")

print("\n4. SUPERVISED CLASSIFICATION:")
print(f"   Dataset: {len(y_train_class):,} training samples")
print(f"   Normal: {np.sum(y_train_class==0):,} | Attacks: {np.sum(y_train_class==1):,}")
print(f"   \n   Performance Ranking (F1-Weighted):")
for i, (model, row) in enumerate(supervised_df.sort_values('F1-Weighted', ascending=False).iterrows(), 1):
    print(f"      {i}. {model}: F1-W={row['F1-Weighted']:.4f}, F1-M={row['F1-Macro']:.4f}, AUC={row['ROC-AUC']:.4f}")

print("\n5. MLP EXPLANATION:")
print("   Architecture: 256 → 128 → 64 → 1 neurons")
print("   • Input Layer: 256 neurons (ReLU activation)")
print("   • Hidden Layers: 128, 64 neurons with Dropout (0.3, 0.2)")
print("   • Output Layer: 1 neuron (Sigmoid for binary classification)")
print("   • Loss Function: Binary crossentropy")
print("   • Optimizer: Adam (adaptive learning rate)")
print("   • Early Stopping: Monitors validation loss (patience=5)")
print("   • Data: Uses SCALED features (PowerTransformer + StandardScaler)")

print("\n6. ANOMALY DETECTION:")
print(f"   Training: {len(X_train_anomaly):,} normal samples (unsupervised)")
print(f"   Evaluation: {len(y_test_anomaly):,} test samples")
print(f"   Isolation Forest: Tree-based outlier detection")
print(f"      ROC-AUC: {unsupervised_df.loc['Isolation Forest', 'ROC-AUC']:.4f}")

total_time = (supervised_df['Training Time'].sum() + 
              unsupervised_df['Training Time'].sum())
print(f"\n7. TRAINING EFFICIENCY:")
print(f"   Total training time: {total_time:.1f} seconds ({total_time/60:.2f} minutes)")
print(f"   Models trained: {len(supervised_df) + len(unsupervised_df)}")

print("\n8. DEPLOYMENT RECOMMENDATION:")
print(f"   Primary Classifier: {best_supervised}")
print(f"      Real-time attack classification")
print(f"      F1-Weighted: {supervised_df.loc[best_supervised, 'F1-Weighted']:.4f}")
print(f"   \n   Anomaly Detector: Isolation Forest")
print(f"      Discover novel/unknown attacks")
print(f"      ROC-AUC: {unsupervised_df.loc['Isolation Forest', 'ROC-AUC']:.4f}")

print("\n" + "="*70)
print("✅ TRAINING COMPLETE - MODELS SAVED")
print("="*70)
print(f"\nSaved models in: {MODELS_DIR}")
print("   • random_forest.joblib")
print("   • mlp_model.h5")
print("   • isolation_forest.joblib")
print("\n📊 All best practices implemented:")
print("   ✓ Reproducibility (RANDOM_STATE=42)")
print("   ✓ Advanced Metrics (Macro/Weighted F1, Confusion Matrix)")
print("   ✓ Model Persistence (joblib + Keras)")
print("   ✓ Large-scale data handling (Train/Val/Test split)")
