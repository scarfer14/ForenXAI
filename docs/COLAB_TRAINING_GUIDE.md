# Google Colab Training Guide

## 📋 Step-by-Step: Local Feature Engineering → Colab Training

### Step 1: Run Feature Engineering Locally ✅

```bash
python src\backend\example_run_feature_prep.py
```

**Output:**
- `data/processed/train_features.csv` (~600 MB)
- `data/processed/validation_features.csv` (~75 MB)
- `data/processed/test_features.csv` (~75 MB)
- `src/backend/feature_engineering/encoders/` (encoders)
- `src/backend/feature_engineering/scalers/` (scalers)

---

### Step 2: Package Everything

```bash
python package_for_colab.py
```

**Creates:** `colab_training_package.zip` (~750 MB)

---

### Step 3: Upload to Google Drive

1. Upload `colab_training_package.zip` to your Google Drive
2. Note the location (e.g., `MyDrive/ForenXAI/`)

---

### Step 4: Google Colab Setup

#### Cell 1: Mount Drive & Extract Package

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Extract package
import zipfile
import os

# Update path to where you uploaded the zip
ZIP_PATH = '/content/drive/MyDrive/colab_training_package.zip'

# Extract
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall('/content/training/')
    
print("✅ Package extracted to /content/training/")
os.listdir('/content/training/')
```

#### Cell 2: Verify Files

```python
import os

# Check engineered features
print("Engineered Features:")
print(f"  Train: {os.path.exists('/content/training/train_features.csv')}")
print(f"  Val: {os.path.exists('/content/training/validation_features.csv')}")
print(f"  Test: {os.path.exists('/content/training/test_features.csv')}")

print("\nFeature Engineering Artifacts:")
print(f"  Encoders: {os.path.exists('/content/training/feature_engineering/encoders')}")
print(f"  Scalers: {os.path.exists('/content/training/feature_engineering/scalers')}")

print("\nNotebook:")
print(f"  train_model.ipynb: {os.path.exists('/content/training/train_model.ipynb')}")
```

#### Cell 3: Load Engineered Features

```python
import pandas as pd

# Load engineered features (fast - already processed!)
df_train = pd.read_csv('/content/training/train_features.csv')
df_val = pd.read_csv('/content/training/validation_features.csv')
df_test = pd.read_csv('/content/training/test_features.csv')

print(f"✅ Train: {df_train.shape}")
print(f"✅ Val: {df_val.shape}")
print(f"✅ Test: {df_test.shape}")
print(f"\nColumns: {df_train.columns.tolist()}")
```

#### Cell 4: Prepare for Training

```python
# Separate features and target
# Assuming 'Label' or 'Attack' is your target column

TARGET_COL = 'Label'  # Change if needed

X_train = df_train.drop(columns=[TARGET_COL, 'Attack'])  # Drop label columns
y_train = df_train[TARGET_COL]

X_val = df_val.drop(columns=[TARGET_COL, 'Attack'])
y_val = df_val[TARGET_COL]

X_test = df_test.drop(columns=[TARGET_COL, 'Attack'])
y_test = df_test[TARGET_COL]

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"\nClass distribution:")
print(y_train.value_counts())
```

#### Cell 5: Train Random Forest (Example)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("Training Random Forest...")

# Train
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf.fit(X_train, y_train)

# Validate
y_val_pred = rf.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)

print(f"\n✅ Validation Accuracy: {val_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))

# Save model
joblib.dump(rf, '/content/training/rf_model.joblib')
print("\n✅ Model saved!")
```

#### Cell 6: Download Trained Model

```python
from google.colab import files

# Download trained model
files.download('/content/training/rf_model.joblib')

print("✅ Model downloaded!")
```

---

## 🚀 Alternative: Upload Features Directly to Drive

Instead of packaging, you can also:

1. **Manually upload to Drive:**
   - Upload `data/processed/` folder to Drive
   - Upload `src/backend/feature_engineering/` folder to Drive
   - Upload `train_model.ipynb` to Drive

2. **Access in Colab:**
```python
# Mount drive
from google.colab import drive
drive.mount('/content/drive')

# Load directly from Drive
import pandas as pd
df_train = pd.read_csv('/content/drive/MyDrive/ForenXAI/data/processed/train_features.csv')
df_val = pd.read_csv('/content/drive/MyDrive/ForenXAI/data/processed/validation_features.csv')
df_test = pd.read_csv('/content/drive/MyDrive/ForenXAI/data/processed/test_features.csv')
```

---

## ⚡ Pro Tips

1. **Enable GPU:** Runtime → Change runtime type → T4 GPU
2. **Keep Session Alive:** Keep browser tab open
3. **Save Frequently:** Save models to Drive after training
4. **Use Sampling:** If too slow, sample data:
   ```python
   df_train = df_train.sample(n=100000, random_state=42)
   ```

---

## 📊 File Sizes (Approximate)

- `train_features.csv`: ~600 MB
- `validation_features.csv`: ~75 MB  
- `test_features.csv`: ~75 MB
- `feature_engineering/`: ~10 MB
- **Total ZIP**: ~750 MB

Upload time: 5-10 minutes on typical connection.

---

## ❓ Troubleshooting

**Issue: "File not found"**
- Check ZIP_PATH in Cell 1
- Verify upload completed

**Issue: "Out of memory"**
- Sample the data: `df_train.sample(n=50000)`
- Use Colab Pro for more RAM

**Issue: "Columns missing"**
- Check feature engineering completed successfully
- Verify all CSVs have same columns (except row count)
