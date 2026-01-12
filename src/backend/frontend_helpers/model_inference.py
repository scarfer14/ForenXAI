"""
Runtime inference utilities for the ForenXAI package.

- load_models(): loads persisted artifacts from ForenXAI/models or a user path
- predict(df, clf, iso): applies models to a prepared DataFrame 

**model_inference.py** is the thin runtime layer between saved artifacts and the UI: 
load_models() finds and deserializes the final severity classifier (the TF‑IDF + 
RandomForest pipeline) and the Isolation Forest, then predict() applies those exact 
trained models to whatever the frontend ingests. 
"""

from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

# Runtime ingestion for frontend uploads is handled in frontend_ingest.py

# Helper to find candidate model directories
def _candidate_model_dirs(explicit: str | None = None) -> list[Path]:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    here = Path(__file__).resolve().parent  # .../ForenXAI/src
    candidates.append(here.parent / 'models')               # ForenXAI/models
    candidates.append(Path.cwd() / 'models')                # ./models (cwd)
    return candidates

# Load models from disk
def load_models(path: str | None = None):
    """
    Load models from the first directory that contains the expected artifacts.
    Expected files:
      - severity_clf.joblib
      - anomaly_iso.joblib
    """
    last_error = None
    for p in _candidate_model_dirs(path):
        try:
            clf = joblib.load(p / 'severity_clf.joblib')
            iso = joblib.load(p / 'anomaly_iso.joblib')
            return clf, iso
        except Exception as e:
            last_error = e
            continue
    raise FileNotFoundError(f"Could not load models from candidates. Last error: {last_error}")

# Apply models to data
def predict(df: pd.DataFrame, clf, iso) -> pd.DataFrame:
    """
    Apply models to a prepared DataFrame or path to CSV.
    Returns a copy with columns:
      - Predicted_Severity
      - Anomaly_Score
      - Anomaly_Flag
    """
    if isinstance(df, str):
        raise TypeError(
            "predict now expects a prepared DataFrame; call frontend_ingest.load_frontend_payload "
            "before invoking inference."
        )

    dfp = df.copy()

    # Severity prediction: expect text-based pipeline (e.g., TF-IDF + tree/linear)
    texts = dfp['Payload_Description'].astype(str).fillna('')
    # TODO: Persist SHAP-ready inputs + MITRE IDs per row so dashboards can show token-level importance beside the mapped ATT&CK technique (Tip: stash the tf-idf vector and mapping output in the returned DataFrame).
    dfp['Predicted_Severity'] = clf.predict(texts)

    # Anomaly detection features: use available simple features safely
    feat_cols = []
    if 'Hour' in dfp.columns:
        feat_cols.append('Hour')
    if 'Src_IP_LastOctet' in dfp.columns:
        feat_cols.append('Src_IP_LastOctet')
    # Fallback if nothing is present
    if not feat_cols:
        dfp['Hour'] = -1
        feat_cols = ['Hour']

    iso_feats = dfp[feat_cols].fillna(-1)
    # decision_function: higher → normal, lower → anomalous
    dfp['Anomaly_Score'] = iso.decision_function(iso_feats)
    dfp['Anomaly_Flag'] = iso.predict(iso_feats)  # -1 anomaly, 1 normal

    return dfp


__all__ = [
    'load_models',
    'predict',
]
