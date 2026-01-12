# ai_dashboard.py
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from frontend_ingest import load_frontend_payload
from model_inference import load_models, predict



''' 
   REFERENCES:
    Streamlit documentation: https://docs.streamlit.io/
    https://www.datacamp.com/tutorial/streamlit (**Note**: DON'T USE ANACONDA ENVIRONMENT)
    SHAP documentation: https://shap.readthedocs.io/en/latest/
'''

# TODO:Study how to connect the backend to the frontend for better integration of model inference and explainability.
# TODO: Study how to connect SHAP explanations with MITRE ATT&CK framework for better context in cybersecurity analysis.
# TODO: Display MITRE ATT&CK tactic/technique context alongside SHAP tokens (Tip: join on backend-provided technique IDs so analysts see both important words and mapped adversary behavior).

# ---------------------------------------------------------------

# PURPOSE: High-level flow of ai_dashboard.py
    # --------------------[Overview Section]------------------------------
    # 1 → User uploads CSV file
    # 2 → Data is validated and prepared for inference
    # 3 → Pretrained models are loaded
    # 4 → Model generates:
    #      → anomaly prediction (binary)
    #      → anomaly / severity score (continuous)
    # 5 → Results table and summary metrics are displayed
    
    # --------------------[Charts Section]------------------------------
    # 6 → Analytical charts are rendered (trend, distribution)

    # --------------------[Explainability Section]------------------
    # 7  → SHAP explains the selected prediction

# ---------------------------------------------------------------

st.set_page_config(page_title="AI Forensics Dashboard", layout="wide")
st.title("🧠 AI-Augmented Cybercrime Forensics")

uploaded = st.file_uploader("Upload payload_timeline.csv", type=['csv'])

model_load_state = st.empty()
clf = None
iso = None
try:
    clf, iso = load_models('models')
    model_load_state.success("Models loaded from ./models")
except Exception:
    model_load_state.warning("Models not found. Please run `python train_model.py` first.")

if uploaded is not None and clf is not None:
    dfp = load_frontend_payload(uploaded)
    st.success("File loaded and prepared.")

    st.subheader("Predictions and Anomaly Scores")
    results = predict(dfp, clf, iso)
    st.dataframe(results[['Timestamp','Source_IP','Payload_Type','Severity','Predicted_Severity','Anomaly_Score','Anomaly_Flag']])

    # Show counts
    col1, col2 = st.columns(2)
    col1.metric("Total payloads", len(results))
    col2.metric("Anomalies detected", int((results['Anomaly_Flag'] == -1).sum()))

    # Allow selecting a row to show SHAP explanation (for tree-based models, TreeExplainer works)
    st.subheader("Explainability (SHAP)")
    idx = st.number_input("Row index to explain", min_value=0, max_value=len(results)-1, value=0)
    row = results.iloc[[idx]]
    st.write(row[['Timestamp','Payload_Type','Payload_Description','Severity','Predicted_Severity','Anomaly_Flag']])

    # Compute SHAP values for text -> we need to get the underlying tfidf transform and the tree model
    # If the classifier is a pipeline: [tfidf, clf], and clf is RandomForest
    if hasattr(clf, 'named_steps'):
        tfidf = clf.named_steps['tfidf']
        model = clf.named_steps['clf'] if 'clf' in clf.named_steps else clf.named_steps[list(clf.named_steps.keys())[-1]]
    else:
        tfidf = None
        model = clf

    if tfidf is not None and hasattr(model, 'estimators_'):
        # Transform text to numeric feature matrix
        X_sparse = tfidf.transform(results['Payload_Description'].astype(str))
        X = X_sparse.toarray()  # convert sparse -> dense

        explainer = shap.TreeExplainer(model)

        # Select the row to explain
        X_row = X[idx].reshape(1, -1)

        shap_values = explainer.shap_values(X_row)

        # Get predicted class
        pred_class = model.predict(X_row)[0]

        # Flatten SHAP values safely
        if isinstance(shap_values, list):
            class_index = list(model.classes_).index(pred_class)
            sv = shap_values[class_index]
        else:
            sv = shap_values

        # Convert to 1D
        if hasattr(sv, "toarray"):
            sv = sv.toarray().ravel()
        else:
            sv = sv.ravel()

        # Feature names
        feature_names = tfidf.get_feature_names_out()

        # Take min length between feature_names and sv
        min_len = min(len(feature_names), len(sv))
        feature_names = feature_names[:min_len]
        sv = sv[:min_len]

        # Top N features
        N = min(10, len(sv))
        top_idx = abs(sv).argsort()[-N:][::-1]

        # Build explanation table
        explanation = [(feature_names[i], float(sv[i])) for i in top_idx]
        st.table(pd.DataFrame(explanation, columns=['token', 'shap_value']))

    else:
        st.info("SHAP explainability not available for this pipeline (requires TF-IDF + tree-based model).")

else:
    st.info("Upload a CSV and ensure models are trained (run `python train_model.py`).")