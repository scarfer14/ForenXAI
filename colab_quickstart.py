"""
Google Colab Quick Setup for ForenXAI
Copy and paste each cell into your Colab notebook
"""

# ============================================================
# CELL 1: Install All Required Packages
# ============================================================
!pip install -q pandas==2.2.3 numpy==2.2.6 scipy==1.16.3 scikit-learn==1.8.0 joblib==1.5.3 tensorflow==2.18.0 matplotlib==3.10.8 shap==0.50.0

print("✅ All packages installed successfully!")

# Verify installations
import pandas as pd
import numpy as np
import scipy
import sklearn
import joblib
import tensorflow as tf
import matplotlib
import shap

print(f"✅ Verification:")
print(f"   • pandas: {pd.__version__}")
print(f"   • numpy: {np.__version__}")
print(f"   • scikit-learn: {sklearn.__version__}")
print(f"   • tensorflow: {tf.__version__}")
print(f"   • shap: {shap.__version__}")


# ============================================================
# CELL 2: Mount Google Drive
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

print("\n✅ Google Drive mounted successfully!")
print("   Models will be saved to: /content/drive/MyDrive/ForenXAI_Models/")
print("   🔗 Access at: https://drive.google.com/drive/folders/1dEjrSobcsnv_uKq4tCgniIOt6uBU0id2")


# ============================================================
# CELL 3: Setup Directory Structure
# ============================================================
import os

# Create necessary directories
base_dir = '/content/drive/MyDrive/ForenXAI'
models_dir = '/content/drive/MyDrive/ForenXAI_Models'

os.makedirs(base_dir, exist_ok=True)
os.makedirs(f'{base_dir}/data/processed', exist_ok=True)
os.makedirs(f'{base_dir}/src/backend', exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

print("✅ Directory structure created:")
print(f"   • Base: {base_dir}")
print(f"   • Data: {base_dir}/data/processed/")
print(f"   • Scripts: {base_dir}/src/backend/")
print(f"   • Models: {models_dir}")


# ============================================================
# CELL 4: Upload Files (MANUAL STEP)
# ============================================================
# Option 1: Use Colab's file upload widget
from google.colab import files

print("📤 Upload your data files now...")
print("\nExpected files:")
print("   • train_features.csv")
print("   • validation_features.csv")
print("   • test_features.csv")
print("   • train_features_dl.csv")
print("   • validation_features_dl.csv")
print("   • test_features_dl.csv")
print("\nAfter upload, move files to: /content/drive/MyDrive/ForenXAI/data/processed/")

# Uncomment to use file upload widget:
# uploaded = files.upload()


# ============================================================
# CELL 5: Verify Data Files Exist
# ============================================================
import os

data_dir = '/content/drive/MyDrive/ForenXAI/data/processed'
required_files = [
    'train_features.csv',
    'validation_features.csv', 
    'test_features.csv',
    'train_features_dl.csv',
    'validation_features_dl.csv',
    'test_features_dl.csv'
]

print("📁 Checking for required data files...")
all_present = True
for file in required_files:
    filepath = os.path.join(data_dir, file)
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024*1024)
        print(f"   ✅ {file} ({size_mb:.2f} MB)")
    else:
        print(f"   ❌ {file} - NOT FOUND")
        all_present = False

if all_present:
    print("\n✅ All data files present! Ready to train.")
else:
    print("\n❌ Missing files! Please upload them to:")
    print(f"   {data_dir}")


# ============================================================
# CELL 6: Upload Training Script
# ============================================================
# Option 1: Upload train_ml_models.py using file upload
from google.colab import files

print("📤 Upload train_ml_models.py now...")
uploaded = files.upload()

# Move to correct location
import shutil
if 'train_ml_models.py' in uploaded:
    shutil.move('train_ml_models.py', '/content/drive/MyDrive/ForenXAI/src/backend/train_ml_models.py')
    print("✅ Training script uploaded successfully!")


# ============================================================
# CELL 7: Run Training
# ============================================================
# Change to script directory
%cd /content/drive/MyDrive/ForenXAI/src/backend

# Run the training script
!python train_ml_models.py


# ============================================================
# CELL 8: Verify Models Were Saved
# ============================================================
import os

models_dir = '/content/drive/MyDrive/ForenXAI_Models'

print("📦 Checking saved models...")
if os.path.exists(models_dir):
    files = sorted(os.listdir(models_dir))
    if files:
        print(f"\n✅ Found {len(files)} files in models directory:")
        for f in files:
            filepath = os.path.join(models_dir, f)
            size_mb = os.path.getsize(filepath) / (1024*1024)
            print(f"   • {f} ({size_mb:.2f} MB)")
        print(f"\n🔗 Access at: https://drive.google.com/drive/folders/1dEjrSobcsnv_uKq4tCgniIOt6uBU0id2")
    else:
        print("❌ Models directory is empty")
else:
    print("❌ Models directory not found")


# ============================================================
# CELL 9: Download Models (Optional)
# ============================================================
# Download models to your local machine
from google.colab import files

models_dir = '/content/drive/MyDrive/ForenXAI_Models'

print("📥 Downloading models...")
for filename in os.listdir(models_dir):
    filepath = os.path.join(models_dir, filename)
    if os.path.isfile(filepath):
        print(f"   Downloading: {filename}")
        files.download(filepath)

print("✅ Download complete!")


# ============================================================
# CELL 10: Test Model Loading (Optional)
# ============================================================
import joblib
from tensorflow.keras.models import load_model

models_dir = '/content/drive/MyDrive/ForenXAI_Models'

# Load Random Forest
rf_path = os.path.join(models_dir, 'random_forest_pipeline.joblib')
if os.path.exists(rf_path):
    rf_model = joblib.load(rf_path)
    print("✅ Random Forest loaded successfully")
else:
    print("❌ Random Forest not found")

# Load MLP
mlp_path = os.path.join(models_dir, 'mlp_model.h5')
if os.path.exists(mlp_path):
    mlp_model = load_model(mlp_path)
    print("✅ MLP model loaded successfully")
else:
    print("❌ MLP model not found")

# Load Isolation Forest
iso_path = os.path.join(models_dir, 'isolation_forest_pipeline.joblib')
if os.path.exists(iso_path):
    iso_model = joblib.load(iso_path)
    print("✅ Isolation Forest loaded successfully")
else:
    print("❌ Isolation Forest not found")

print("\n🎉 All models loaded and ready for inference!")
