
## ForenXAI

AI-powered network intrusion detection system that processes network traffic data, trains supervised and unsupervised ML models, and delivers explainable predictions through a Streamlit dashboard. The project uses the NF-CSE-CIC-IDS2018 dataset and augments it with CTGAN-generated synthetic samples for improved model generalization.

### PROJECT PHASES

**Phase 1: Synthetic Data Generation**
- **Script**: `src/backend/synthetic_data_generation.ipynb`
- **Technology**: CTGAN (Conditional Tabular GAN) for generating realistic synthetic network traffic
- **Output**: ≥5k synthetic samples saved to `data/synthetic/` for data augmentation
- **Purpose**: Enhance training data diversity and address class imbalance

**Phase 2: Data Preparation & Feature Engineering**
- **Scripts**: 
  - `src/backend/feature_prep.py` - Production-grade feature engineering pipeline
  - `src/backend/example_run_feature_prep.py` - Orchestrator script
- **Process**:
  - Combines real + synthetic datasets (~1.8M training samples)
  - 70/15/15 train/validation/test split
  - Categorical encoding (PROTOCOL, L7_PROTO → numeric)
  - Feature scaling: PowerTransformer (Yeo-Johnson) + StandardScaler for deep learning
  - Unscaled features preserved for tree-based models
- **Output**: 36 engineered features saved to `data/processed/`
  - `train/val/test_features.csv` (unscaled for Random Forest, Isolation Forest)
  - `train/val/test_features_dl.csv` (scaled for MLP)

**Phase 3: Model Training with ML Best Practices**
- **Script**: `src/backend/train_ml_models.py`
- **Models**:
  1. **Random Forest** (Supervised Classification)
     - Hyperparameter tuning via RandomizedSearchCV (20 combinations, 3-fold CV)
     - Parameters: n_estimators, max_depth, min_samples_split/leaf, max_features
     - Wrapped in sklearn Pipeline for production deployment
  2. **MLP - Multi-Layer Perceptron** (Deep Learning Classification)
     - Architecture: 256 → 128 → 64 → 1 (with dropout layers)
     - Uses scaled features (PowerTransformer + StandardScaler)
     - Early stopping to prevent overfitting
  3. **Isolation Forest** (Unsupervised Anomaly Detection)
     - Discovers novel/unknown attack patterns
     - Wrapped in sklearn Pipeline
- **Best Practices Implemented**:
  - ✅ Cross-validation verification (split balance checking)
  - ✅ Hyperparameter tuning (RandomizedSearchCV)
  - ✅ Pipeline wrapping (production-ready deployment)
  - ✅ Reproducibility (RANDOM_STATE=42, TensorFlow/NumPy seeding)
  - ✅ Comprehensive metrics (F1-Binary/Macro/Weighted, ROC-AUC, Confusion Matrix)
- **Output**: Trained pipelines saved to `models/`
  - `random_forest_pipeline.joblib`
  - `mlp_model.h5`
  - `isolation_forest_pipeline.joblib`

**Phase 4: Model Evaluation**
- **Script**: `src/backend/model_evaluation.ipynb`
- **Metrics**: Accuracy, Precision, Recall, F1-Score (Binary/Macro/Weighted), ROC-AUC
- **Visualizations**: Confusion matrices, performance comparison tables
- **Evaluation**: 15% hold-out test set (~230K samples)

**Phase 5: Inference & Dashboard**
- **Backend**: `src/backend/model_inference.py` - Loads trained models and generates predictions
- **Frontend**: `src/frontend/ai_dashboard.py` - Streamlit interface for real-time detection
- **Features**: Upload network traffic CSVs, view predictions, model explanations

### QUICK START

1. **Environment Setup**
   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   
   # Install dependencies
   pip install -r requirements.txt
   ```
   See `docs/ENVIRONMENT_SETUP.md` for detailed OS-specific instructions.

2. **Data Preparation**
   - Place NF-CSE-CIC-IDS2018 dataset in `data/real/` directory
   - Required files: `train_real.csv`, `val_real.csv`, `test_real.csv`
   - (Optional) Generate synthetic data using CTGAN: Run `synthetic_data_generation.ipynb`

3. **Feature Engineering**
   ```bash
   cd src/backend
   python example_run_feature_prep.py
   ```
   This creates processed features in `data/processed/` (both scaled and unscaled versions).

4. **Model Training**
   ```bash
   cd src/backend
   python train_ml_models.py
   ```
   Training includes:
   - Random Forest hyperparameter tuning (~5-10 minutes)
   - MLP training with early stopping (~10-15 minutes)
   - Isolation Forest fitting (~2-3 minutes)
   - All models saved to `models/` directory

5. **Launch Dashboard**
   ```bash
   streamlit run src/frontend/ai_dashboard.py
   ```
   Upload network traffic CSVs for real-time intrusion detection.

### KEY OUTPUTS

**Trained Models** (in `models/` directory):
- `random_forest_pipeline.joblib` - Production-ready RF classifier with preprocessing
- `mlp_model.h5` - Deep learning MLP model (Keras/TensorFlow format)
- `isolation_forest_pipeline.joblib` - Unsupervised anomaly detector pipeline

**Processed Features** (in `data/processed/` directory):
- `train_features.csv`, `validation_features.csv`, `test_features.csv` - Unscaled (for tree models)
- `train_features_dl.csv`, `validation_features_dl.csv`, `test_features_dl.csv` - Scaled (for MLP)

**Feature Engineering Artifacts** (in `src/backend/feature_engineering/` directory):
- `encoders/label_encoders.joblib` - Categorical encoders (PROTOCOL, L7_PROTO)
- `scalers/power_transformer.joblib` - PowerTransformer for distribution normalization
- `scalers/scaler.joblib` - StandardScaler for final scaling

### TECHNICAL SPECIFICATIONS

**Dataset**: NF-CSE-CIC-IDS2018
- Training: ~1.8M samples
- Validation: ~230K samples  
- Test: ~230K samples
- Features: 36 engineered network traffic features

**Models Performance** (Expected on test set):
- Random Forest: F1-Score ≥ 0.95, ROC-AUC ≥ 0.98
- MLP: F1-Score ≥ 0.94, ROC-AUC ≥ 0.97
- Isolation Forest: Unsupervised outlier detection (no ground truth required)

**Production Features**:
- Modular architecture (decoupled preprocessing, training, inference)
- Pipeline-wrapped models (single artifact deployment)
- Reproducible results (seeded random states)
- Scalable to API/batch deployments beyond Streamlit
