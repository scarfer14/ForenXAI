#!/bin/bash
# ============================================================
# Google Colab Installation Script for ForenXAI
# ============================================================
# Copy and paste these commands into your Google Colab notebook
# Run in a single cell for complete installation

# Core Data Science Libraries
pip install -q pandas==2.2.3
pip install -q numpy==2.2.6
pip install -q scipy==1.16.3

# Machine Learning
pip install -q scikit-learn==1.8.0
pip install -q joblib==1.5.3

# Deep Learning
pip install -q tensorflow==2.18.0

# Visualization
pip install -q matplotlib==3.10.8

# Explainable AI
pip install -q shap==0.50.0

echo "✅ All packages installed successfully!"

# ============================================================
# Alternative: Single Line Installation (Recommended for Colab)
# ============================================================
# pip install -q pandas==2.2.3 numpy==2.2.6 scipy==1.16.3 scikit-learn==1.8.0 joblib==1.5.3 tensorflow==2.18.0 matplotlib==3.10.8 shap==0.50.0

# ============================================================
# Verification
# ============================================================
# python -c "import pandas, numpy, scipy, sklearn, joblib, tensorflow, matplotlib, shap; print('✅ All imports successful!')"
