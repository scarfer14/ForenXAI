# ForenXAI - Google Colab Installation Commands

## 🚀 Quick Install (Copy-Paste into Colab)

### Single Command Installation
```python
!pip install -q pandas==2.2.3 numpy==2.2.6 scipy==1.16.3 scikit-learn==1.8.0 joblib==1.5.3 tensorflow==2.18.0 matplotlib==3.10.8 shap==0.50.0
```

### Individual Package Installation
If you prefer to install packages one by one:

```bash
# Core Data Science
pip install pandas==2.2.3
pip install numpy==2.2.6
pip install scipy==1.16.3

# Machine Learning
pip install scikit-learn==1.8.0
pip install joblib==1.5.3

# Deep Learning
pip install tensorflow==2.18.0

# Visualization
pip install matplotlib==3.10.8

# Explainable AI (XAI)
pip install shap==0.50.0
```

## 📦 Package Versions

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.2.3 | Data manipulation |
| numpy | 2.2.6 | Numerical computing |
| scipy | 1.16.3 | Scientific computing |
| scikit-learn | 1.8.0 | ML algorithms (RF, Isolation Forest) |
| joblib | 1.5.3 | Model serialization |
| tensorflow | 2.18.0 | Deep learning (MLP) |
| matplotlib | 3.10.8 | Data visualization |
| shap | 0.50.0 | Explainable AI |

## 🔧 Installation Notes

### Pre-installed in Colab
These packages come pre-installed in Google Colab:
- ✅ pandas
- ✅ numpy
- ✅ scipy
- ✅ scikit-learn
- ✅ matplotlib
- ✅ tensorflow

**BUT**: We recommend installing specific versions to ensure compatibility.

### Need to Install
These packages are NOT pre-installed:
- ❌ shap (must install)
- ❌ joblib specific version (may differ)

## 🎯 Minimal Installation (if Colab packages work)

If you want to use pre-installed Colab packages, only install:

```python
!pip install -q shap==0.50.0
```

**Warning**: This may cause version conflicts. Recommended to install all packages with specific versions.

## ✅ Verify Installation

After installation, run this to verify:

```python
import pandas as pd
import numpy as np
import scipy
import sklearn
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import shap

print("✅ All packages imported successfully!")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"tensorflow: {tf.__version__}")
print(f"shap: {shap.__version__}")
```

## 🐛 Troubleshooting

### Issue: PackageNotFoundError
**Solution**: 
```python
!pip install --upgrade pip
# Then re-run installation
```

### Issue: Version Conflicts
**Solution**:
```python
# Restart runtime: Runtime → Restart runtime
# Then re-run installation in fresh session
```

### Issue: Import Error
**Solution**:
```python
# Restart runtime after installation
# Runtime → Restart runtime
```

## 🔗 Model Save Location

Models will be automatically saved to:
- **Google Drive**: `/content/drive/MyDrive/ForenXAI_Models/`
- **Direct Link**: https://drive.google.com/drive/folders/1dEjrSobcsnv_uKq4tCgniIOt6uBU0id2

## 📱 Complete Colab Workflow

```python
# 1. Install packages
!pip install -q pandas==2.2.3 numpy==2.2.6 scipy==1.16.3 scikit-learn==1.8.0 joblib==1.5.3 tensorflow==2.18.0 matplotlib==3.10.8 shap==0.50.0

# 2. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 3. Run training
%cd /content/drive/MyDrive/ForenXAI/src/backend
!python train_ml_models.py

# 4. Check saved models
!ls -lh /content/drive/MyDrive/ForenXAI_Models/
```

## 💡 Pro Tips

1. **GPU Acceleration**: Enable GPU for faster training
   - Runtime → Change runtime type → GPU

2. **Longer Sessions**: Colab Pro provides longer runtime
   - Free tier: ~12 hours
   - Pro tier: ~24 hours

3. **Save Checkpoints**: Code automatically saves to Google Drive
   - Won't lose progress if session disconnects

4. **Monitor RAM**: Check RAM usage in top-right corner
   - If running out: reduce `shap_sample_size` in code

## 📚 Additional Resources

- Colab Quickstart: See `colab_quickstart.py`
- Detailed Setup: See `COLAB_SETUP.md`
- Original Requirements: See `requirements.txt`

## ⏱️ Expected Installation Time

- Package installation: ~2-3 minutes
- First import (compilation): ~1-2 minutes
- Total: ~3-5 minutes

## 🎉 Ready to Go!

After installation, you're ready to train ForenXAI models in Google Colab!
