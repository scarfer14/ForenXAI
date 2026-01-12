
## ForenXAI

AI-assisted cyber forensics pipeline that ingests payload timelines, trains severity and anomaly detectors, and serves explainable results through a Streamlit dashboard. The project centers on the NF-CSE-CIC-IDS2018 dataset and augments it with Gretel ACTGAN synthetic samples.

### CORE COMPONENTS

- **Synthetic Data (Phase 1)**: `src/backend/ml_pipeline/synthetic_data_generation.ipynb` installs Gretel tooling, trains an ACTGAN model, and exports ≥5k synthetic rows.

- **Data Prep & Features (Phases 2–3)**: `data_prep.ipynb` validates the cleaned dataset and splits 70/15/15; `feature_engineering.ipynb` builds TF-IDF vocabularies for severity, numeric features 
(Hour, Src_IP_LastOctet) for Isolation Forest, and MITRE ATT&CK tags mapped from payload text/types.

- **Training (Phase 4)**: `train_model.ipynb` locks the severity pipeline as `Pipeline([('tfidf', TfidfVectorizer), ('clf', RandomForestClassifier)])`, tunes it, fits an Isolation Forest on engineered numeric features, and persists both models plus SHAP explainers and MITRE lookup tables with `joblib`.

- **Evaluation (Phase 5)**: `model_evaluation.ipynb` reports accuracy/precision/recall/F1/ROC-AUC, confusion matrices, anomaly detection metrics, SHAP summaries, and MITRE coverage counts on the 15% hold-out set.

- **Runtime & Frontend (Phase 6)**: `src/backend/frontend_helpers/frontend_ingest.py` cleans uploads and derives runtime features; `model_inference.py` loads the saved models and produces predictions; `src/frontend/ai_dashboard.py` surfaces results, SHAP insights, and ATT&CK context.

### QUICK START

1. **Environment**: Create/activate the project venv and install `requirements.txt` (plus optional extras as needed). See `docs/ENVIRONMENT_SETUP.md` for OS-specific steps.

2. **Data**: Place the NF-CSE-CIC-IDS2018 CSV under your preferred path; notebooks read from the absolute path stored in `synthetic_data_generation.ipynb` (update it if needed).

3. **Run Notebooks**: Execute Phases 1–5 in order, ensuring each TODO checklist item is satisfied (schema validation, feature serialization, SHAP artifacts, MITRE mapping, etc.).

4. **Serve Dashboard**: After saving `severity_clf.joblib`, `anomaly_iso.joblib`, TF-IDF vocab, SHAP background, and MITRE mappings under `models/`, launch `streamlit run src/frontend/ai_dashboard.py` and upload payload CSVs for scoring.

### Key Outputs

- `models/severity_clf.joblib`: TF-IDF + RandomForest severity pipeline.
- `models/anomaly_iso.joblib`: Isolation Forest trained on Hour/Src_IP_LastOctet (plus optional engineered fields).
- `models/shap_background.pkl` (or similar): cached data for fast SHAP explanations.
- `models/mitre_lookup.json`: token/type → MITRE tactic/technique mapping consumed by both inference and UI layers.

Keep ingestion, inference, and dashboard modules decoupled so the same models can power APIs or batch jobs beyond Streamlit.
