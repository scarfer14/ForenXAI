
**TITLE**: Machine Learning Pipeline Overview

**[PHASE 1] Synthetic Data Generation (synthetic_data_generation.ipynb)**

- Install/verify `gretel-client` and `gretel-trainer` inside the notebook kernel.
- Load the cleaned NF-CSE-CIC-IDS2018 CSV and keep a lightweight schema dict for validation.
- Configure the ACTGAN trainer (`models.GretelACTGAN`) and capture key hyperparameters.
- Train the generator and export ≥5k synthetic rows for downstream phases.

---------------------------------------------------------------------------------------------------
**[PHASE 2] Data Preprocessing (data_prep.ipynb)**

- Load the cleaned dataset (see DATASET_SOURCE.md) and log shapes/types.
- Sanity-check nulls, duplicates, and basic distributions; visualize anomalies when found.
- Produce the deterministic 70/15/15 split (store random seed) for training, validation, and testing artifacts (training 70%, validation 15%, testing 15%).

---------------------------------------------------------------------------------------------------
**[PHASE 3] Feature Engineering (feature_engineering.ipynb)**

- Track dropped/kept raw columns and justify each decision.
- Build the TF-IDF vocabulary (or other encoders) for the severity pipeline and persist it for inference/SHAP parity.
- Engineer numeric features required by Isolation Forest (`Hour`, `Src_IP_LastOctet`, optional deltas) and document imputations/scaling.
- Attach MITRE ATT&CK tactic/technique tags derived from payload text/type using a maintained lookup table.
- Persist all preprocessing objects (encoders, scalers, MITRE mapping) alongside metadata for later notebooks.

---------------------------------------------------------------------------------------------------
**[PHASE 4] Model Training (train_model.ipynb)**

- Lock the severity classifier as `Pipeline([('tfidf', TfidfVectorizer), ('clf', RandomForestClassifier)])` and tune it via GridSearchCV/RandomizedSearchCV.
- Fit the Isolation Forest on the engineered numeric feature frame, capturing any scaler/imputer used before persistence.
- Record validation (15%) metrics—accuracy, precision, recall, F1, ROC-AUC—for both baseline and tuned models.
- Persist the chosen models, preprocessing artifacts, SHAP background samples, and MITRE lookup files with `joblib`/`json`.

---------------------------------------------------------------------------------------------------
**[PHASE 5] Model Evaluation (model_evaluation.ipynb)**

- Score the locked models on the 15% hold-out split and report accuracy/precision/recall/F1/ROC-AUC.
- Visualize severity performance (confusion matrix, ROC curve) and anomaly detection histograms/threshold behavior.
- Generate SHAP summary plots for the severity classifier and highlight top tokens per class.
- Quantify MITRE ATT&CK coverage by counting predictions per tactic/technique and flagging gaps for future data needs.

---------------------------------------------------------------------------------------------------
**[PHASE 6] Frontend & Runtime Integration (frontend_ingest.py, model_inference.py, ai_dashboard.py)**

- `frontend_ingest.py`: sanitize uploads, canonicalize payload metadata, derive runtime features (Hour, Src_IP_LastOctet), and attach MITRE hints for inference.
- `model_inference.py`: load persisted severity/anomaly models plus SHAP/MITRE artifacts, then output predictions, anomaly flags, and context fields for the UI.
- `ai_dashboard.py`: Streamlit app for file upload, metrics, explainability (SHAP + MITRE context), and anomaly monitoring.

---------------------------------------------------------------------------------------------------

**RESOURCE**: https://www.ibm.com/think/topics/machine-learning-pipeline