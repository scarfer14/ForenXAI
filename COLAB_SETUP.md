# Google Colab Setup Guide for ForenXAI

## 📋 Prerequisites
1. Google Colab account
2. Google Drive access
3. Trained models will be saved to: https://drive.google.com/drive/folders/1dEjrSobcsnv_uKq4tCgniIOt6uBU0id2

## 🚀 Step-by-Step Setup

### 1. Create New Colab Notebook
Go to: https://colab.research.google.com/

### 2. Install Required Packages

Copy and paste this into the first cell of your Colab notebook:

```python
# Install all required packages for ForenXAI
!pip install -q pandas==2.2.3
!pip install -q numpy==2.2.6
!pip install -q scipy==1.16.3
!pip install -q scikit-learn==1.8.0
!pip install -q joblib==1.5.3
!pip install -q matplotlib==3.10.8
!pip install -q tensorflow==2.18.0
!pip install -q shap==0.50.0

print("✅ All packages installed successfully!")
```

### 3. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4. Setup Directory Structure

```python
import os

# Create ForenXAI folder in Google Drive
base_dir = '/content/drive/MyDrive/ForenXAI'
os.makedirs(base_dir, exist_ok=True)
os.makedirs(f'{base_dir}/data/processed', exist_ok=True)
os.makedirs(f'{base_dir}/src/backend', exist_ok=True)

print(f"✅ Directory structure created at: {base_dir}")
```

### 5. Upload Your Data

Upload your processed data files to:
`/content/drive/MyDrive/ForenXAI/data/processed/`

Required files:
- `train_features.csv`
- `validation_features.csv`
- `test_features.csv`
- `train_features_dl.csv`
- `validation_features_dl.csv`
- `test_features_dl.csv`

### 6. Upload Training Script

Upload `train_ml_models.py` to:
`/content/drive/MyDrive/ForenXAI/src/backend/`

### 7. Run Training

```python
# Change to script directory
%cd /content/drive/MyDrive/ForenXAI/src/backend

# Run the training script
!python train_ml_models.py
```

### 8. Access Your Models

After training completes, all models will be saved to:
`/content/drive/MyDrive/ForenXAI_Models/`

Or access directly via:
https://drive.google.com/drive/folders/1dEjrSobcsnv_uKq4tCgniIOt6uBU0id2

## 📦 Output Files

The following files will be saved to Google Drive:

### Models
- `random_forest_pipeline.joblib` - Random Forest classifier
- `mlp_model.h5` - Deep learning MLP model
- `isolation_forest_pipeline.joblib` - Anomaly detector

### Analysis Files
- `feature_importance_rf.csv` - Feature importance scores
- `feature_importance_plot.png` - Feature importance visualization
- `shap_importance_rf.csv` - SHAP values for Random Forest
- `shap_importance_mlp.csv` - SHAP values for MLP
- `shap_summary_random_forest.png` - SHAP visualization (RF)
- `shap_summary_mlp.png` - SHAP visualization (MLP)

## ⚡ Quick Start Template

Here's a complete Colab notebook template:

```python
# ========================================
# Cell 1: Install Packages
# ========================================
!pip install -q pandas numpy scipy scikit-learn joblib matplotlib tensorflow shap
print("✅ Installation complete!")

# ========================================
# Cell 2: Mount Google Drive
# ========================================
from google.colab import drive
drive.mount('/content/drive')

# ========================================
# Cell 3: Setup Directories
# ========================================
import os
base_dir = '/content/drive/MyDrive/ForenXAI'
os.makedirs(f'{base_dir}/data/processed', exist_ok=True)
os.makedirs(f'{base_dir}/src/backend', exist_ok=True)
print(f"✅ Ready at: {base_dir}")

# ========================================
# Cell 4: Upload Files (Manual Step)
# ========================================
# Use Colab's file upload or copy from Drive
# 1. Upload data files to: {base_dir}/data/processed/
# 2. Upload train_ml_models.py to: {base_dir}/src/backend/

# ========================================
# Cell 5: Run Training
# ========================================
%cd /content/drive/MyDrive/ForenXAI/src/backend
!python train_ml_models.py

# ========================================
# Cell 6: Verify Models Saved
# ========================================
import os
models_dir = '/content/drive/MyDrive/ForenXAI_Models'
if os.path.exists(models_dir):
    files = os.listdir(models_dir)
    print(f"✅ Found {len(files)} files in models directory:")
    for f in files:
        print(f"   • {f}")
else:
    print("❌ Models directory not found")
```

## 🔧 Troubleshooting

### Issue: Module not found
**Solution:** Re-run the pip install cell

### Issue: Out of memory
**Solution:** 
- Use Colab Pro for more RAM
- Reduce `shap_sample_size` in the code (line ~605)
- Reduce `mlp_shap_sample_size` (line ~630)

### Issue: Files not found
**Solution:** Verify file paths in Google Drive exactly match:
- Data: `/content/drive/MyDrive/ForenXAI/data/processed/`
- Script: `/content/drive/MyDrive/ForenXAI/src/backend/`

### Issue: Training too slow
**Solution:** 
- Enable GPU: Runtime → Change runtime type → GPU
- Reduce hyperparameter search iterations (line ~370: `n_trials = 10` → `n_trials = 5`)
- Reduce RF search iterations (line ~265: `n_iter=20` → `n_iter=10`)

## 📊 Expected Runtime

On Colab with GPU:
- Data Loading: ~2-5 minutes
- Random Forest Training: ~10-20 minutes
- MLP Training: ~5-10 minutes
- Isolation Forest: ~2-5 minutes
- SHAP Analysis: ~5-10 minutes
- **Total: ~25-50 minutes**

## 🎯 Next Steps

After training:
1. Download models from Google Drive
2. Use models for inference/prediction
3. Share SHAP visualizations for reporting
4. Deploy models to production

## 📞 Support

For issues, check:
- File paths are correct
- All packages installed
- Google Drive has sufficient space (need ~500MB-1GB)
- Colab runtime is still active
