import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, accuracy_score, 
    precision_score, recall_score
)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import time
from io import BytesIO
import base64
import os
import glob
import joblib

# ================================================================
# PAGE CONFIGURATION
# ================================================================
st.set_page_config(
    page_title="AI Model Evaluation Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #10b981;
    }
    .warning-metric {
        border-left-color: #f59e0b;
    }
    .error-metric {
        border-left-color: #ef4444;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# HELPER FUNCTIONS
# ================================================================

def get_metric_color(value, metric_type='accuracy'):
    """Return color based on metric value and thresholds"""
    if metric_type in ['accuracy', 'precision', 'recall', 'f1']:
        if value >= 0.9:
            return "#10b981"  # Green
        elif value >= 0.7:
            return "#f59e0b"  # Yellow
        else:
            return "#ef4444"  # Red
    return "#3b82f6"  # Blue default

def calculate_classification_metrics(y_true, y_pred, y_proba=None):
    """Calculate comprehensive classification metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    # Per-class metrics
    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics['per_class_metrics'] = class_report
    
    # ROC data if probabilities are available
    if y_proba is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        metrics['roc_data'] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
    
    return metrics

def create_confusion_matrix_heatmap(cm, labels, normalize=False):
    """Create an interactive confusion matrix heatmap"""
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_norm
        text_display = [[f'{val:.2%}<br>({cm[i][j]})' for j, val in enumerate(row)] 
                       for i, row in enumerate(cm_norm)]
    else:
        cm_display = cm
        text_display = cm
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_display,
        x=[f'Predicted: {label}' for label in labels],
        y=[f'Actual: {label}' for label in labels],
        text=text_display,
        texttemplate='%{text}',
        colorscale='YlOrRd',
        showscale=True
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='Actual Label',
        height=500,
        width=600
    )
    
    return fig

def create_roc_curve(fpr, tpr, roc_auc):
    """Create ROC curve visualization"""
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color='#3b82f6', width=3),
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
    ))
    
    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'ROC Curve (AUC = {roc_auc:.3f})',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        width=600,
        hovermode='closest',
        showlegend=True
    )
    
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    
    return fig

def export_to_csv(df):
    """Export dataframe to CSV"""
    return df.to_csv(index=False).encode('utf-8')

def export_to_json(data):
    """Export data to JSON"""
    return json.dumps(data, indent=2).encode('utf-8')

# ================================================================
# MAIN APPLICATION
# ================================================================

st.title("ForenXAI")
st.markdown("*Comprehensive ML Model Performance Analysis*")

# ================================================================
# SIDEBAR - MODEL SELECTION & CONFIGURATION
# ================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Dataset Upload Section (moved above model selection)
    st.subheader("1️⃣ Dataset Upload")

    uploaded_file = st.file_uploader(
        "Upload Test Dataset",
        type=['csv', 'json', 'xlsx'],
        help="Upload your test dataset with actual labels and predictions"
    )

    st.divider()

    # Model Selection Section
    st.subheader("2️⃣ Model Selection")
    
    # Discover available .joblib models in likely folders (case variations)
    base_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(base_dir)
    grandparent_dir = os.path.dirname(parent_dir)

    # Look in likely locations relative to this file and up the tree
    candidates = []
    # local folders inside frontend
    candidates += [os.path.join(base_dir, d) for d in ('models', 'Models', 'model', 'Model')]
    # sibling folders under src (e.g., src/Models)
    candidates += [os.path.join(parent_dir, d) for d in ('models', 'Models')]
    # project-level folders (in case file layout differs)
    candidates += [os.path.join(grandparent_dir, d) for d in ('models', 'Models')]

    model_files = []
    for d in candidates:
        if os.path.isdir(d):
            model_files.extend(glob.glob(os.path.join(d, '*.joblib')))
    # Deduplicate by basename (filename) so the same model saved in multiple
    # locations doesn't appear multiple times in the selector.
    unique_by_name = {}
    for p in model_files:
        name = os.path.basename(p).lower()
        if name not in unique_by_name:
            unique_by_name[name] = p
    model_files = sorted(unique_by_name.values())
    # Exclude the severity classifier (case insensitive)
    model_files = [p for p in model_files if os.path.basename(p).lower() != 'severity_clf.joblib']

    available_models = []
    for idx, path in enumerate(model_files):
        fname = os.path.basename(path)
        stem = os.path.splitext(fname)[0]
        # Try to infer lightweight metadata (file timestamp). Avoid loading heavy objects here.
        try:
            mtime = os.path.getmtime(path)
            training_date = datetime.fromtimestamp(mtime).isoformat()
        except Exception:
            training_date = 'N/A'

        available_models.append({
            'id': f'model-{idx+1:03d}',
            'name': stem,
            'version': 'unknown',
            'type': 'Classification',
            'training_date': training_date,
            'accuracy': 0.0,
            'path': path
        })

    if not available_models:
        st.info('No model files found in the models/ folder (severity_clf.joblib excluded).')

    model_options = [f"{m['name']} (v{m['version']})" for m in available_models]
    if available_models:
        selected_model_idx = st.selectbox(
            "Select Model",
            range(len(model_options)),
            format_func=lambda x: model_options[x]
        )
        selected_model = available_models[selected_model_idx]
    else:
        selected_model = None
    
    # Model Info Card
    with st.expander("📊 Model Details", expanded=True):
        if selected_model:
            st.markdown(f"""
            **Model ID:** `{selected_model['id']}`  
            **Type:** {selected_model['type']}  
            **Training Date:** {selected_model['training_date']}  
            **Base Accuracy:** {selected_model['accuracy']:.2%}
            """)
        else:
            st.write("No model available in models/ (severity_clf.joblib excluded).")
    
    # Display options
    st.divider()
    st.subheader("3️⃣ Display Options")
    
    show_normalized_cm = st.checkbox("Show Normalized Confusion Matrix", value=False)
    show_per_class = st.checkbox("Show Per-Class Metrics", value=True)
    results_per_page = st.slider("Predictions per page", 10, 100, 50)
    
    st.divider()
    
    # Export Section
    st.subheader("4️⃣ Export Options")
    export_format = st.selectbox(
        "Export Format",
        ["CSV", "JSON", "PDF Report"]
    )

# ================================================================
# MAIN CONTENT
# ================================================================

if uploaded_file is not None:
    try:
        # Load dataset based on file type
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ Dataset loaded successfully: {len(df)} samples")
        
        # ================================================================
        # DATASET PREVIEW & STATISTICS
        # ================================================================
        with st.expander("📋 Dataset Preview & Statistics", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Samples", len(df))
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                missing_values = df.isnull().sum().sum()
                st.metric("Missing Values", missing_values)
            with col4:
                unique_classes = df['actual'].nunique() if 'actual' in df.columns else 0
                st.metric("Unique Classes", unique_classes)
            
            st.subheader("First 20 Rows")
            st.dataframe(df.head(20), use_container_width=True)
            
            st.subheader("Dataset Info")
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.write("**Data Types:**")
                st.dataframe(df.dtypes.to_frame('Type'), use_container_width=True)
            
            with col_info2:
                st.write("**Missing Values by Column:**")
                missing_df = df.isnull().sum().to_frame('Missing Count')
                missing_df['Percentage'] = (missing_df['Missing Count'] / len(df) * 100).round(2)
                st.dataframe(missing_df, use_container_width=True)
        
        # Validate required columns
        required_cols = ['actual', 'predicted']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ Dataset must contain columns: {required_cols}")
            st.info(f"Found columns: {list(df.columns)}")
            st.stop()

        # Optionally run the selected model to generate predictions (overrides uploaded 'predicted')
        use_model = False
        if selected_model is not None:
            use_model = st.checkbox("Run selected model on uploaded data (overrides 'predicted')", value=False)

        if use_model:
            model_path = selected_model.get('path') if selected_model else None
            if not model_path or not os.path.exists(model_path):
                st.error('Selected model file not found.')
                st.stop()

            try:
                with st.spinner(f"Loading model {selected_model['name']}..."):
                    model = joblib.load(model_path)

                # Prepare feature matrix: prefer model.feature_names_in_ when available
                prob_cols = [col for col in df.columns if col.startswith('prob_')]
                exclude_cols = ['actual', 'predicted'] + prob_cols

                feature_cols = None
                try:
                    if hasattr(model, 'feature_names_in_'):
                        feature_cols = [c for c in list(model.feature_names_in_) if c in df.columns]
                    elif hasattr(model, 'named_steps'):
                        # If a pipeline was saved, try the final estimator
                        final = list(model.named_steps.values())[-1]
                        if hasattr(final, 'feature_names_in_'):
                            feature_cols = [c for c in list(final.feature_names_in_) if c in df.columns]
                except Exception:
                    feature_cols = None

                if not feature_cols:
                    feature_cols = [c for c in df.columns if c not in exclude_cols]

                X = df[feature_cols].copy()
                # Select numeric columns; if none, attempt coercion
                X_numeric = X.select_dtypes(include=[np.number])
                if X_numeric.shape[1] == 0:
                    X_conv = X.apply(pd.to_numeric, errors='coerce')
                    # drop all-empty numeric columns
                    X_conv = X_conv.loc[:, X_conv.notna().any(axis=0)]
                    if X_conv.shape[1] == 0:
                        st.error('No numeric feature columns detected for model prediction. Ensure features are numeric or provide features matching the model.')
                        st.stop()
                    else:
                        X_numeric = X_conv.fillna(0)

                try:
                    preds = model.predict(X_numeric)
                    df['predicted'] = preds

                    # Add probability columns if available
                    if hasattr(model, 'predict_proba'):
                        try:
                            proba = model.predict_proba(X_numeric)
                            n_classes = proba.shape[1]
                            if n_classes == 2:
                                df['prob_0'] = proba[:, 0]
                                df['prob_1'] = proba[:, 1]
                            else:
                                for i in range(n_classes):
                                    df[f'prob_{i}'] = proba[:, i]
                        except Exception:
                            # If predict_proba fails, skip probability columns
                            pass

                    st.success('Model predictions added to dataset and will be used for evaluation.')
                except Exception as e:
                    st.error(f'Failed to run model prediction: {e}')
                    st.stop()
            except Exception as e:
                st.error(f'Failed to load/run model: {e}')
                st.stop()
        
        # Calculate metrics
        y_true = df['actual']
        y_pred = df['predicted']
        
        # Check if probability columns exist
        prob_cols = [col for col in df.columns if col.startswith('prob_')]
        if prob_cols and len(np.unique(y_true)) == 2:
            y_proba = df[prob_cols].values
        else:
            y_proba = None
        
        # Calculate all metrics
        with st.spinner("🔄 Calculating evaluation metrics..."):
            time.sleep(0.5)  # Simulate processing
            metrics = calculate_classification_metrics(y_true, y_pred, y_proba)
        
        # ================================================================
        # MAIN METRICS DASHBOARD
        # ================================================================
        st.header("📊 Evaluation Metrics Dashboard")
        
        # Main metric cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            acc_color = get_metric_color(metrics['accuracy'], 'accuracy')
            st.markdown(f"""
            <div class="metric-card {'success-metric' if metrics['accuracy'] >= 0.9 else 'warning-metric' if metrics['accuracy'] >= 0.7 else 'error-metric'}">
                <h4>Accuracy</h4>
                <h2 style="color: {acc_color};">{metrics['accuracy']:.2%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            prec_color = get_metric_color(metrics['precision'], 'precision')
            st.markdown(f"""
            <div class="metric-card {'success-metric' if metrics['precision'] >= 0.9 else 'warning-metric' if metrics['precision'] >= 0.7 else 'error-metric'}">
                <h4>Precision</h4>
                <h2 style="color: {prec_color};">{metrics['precision']:.2%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            rec_color = get_metric_color(metrics['recall'], 'recall')
            st.markdown(f"""
            <div class="metric-card {'success-metric' if metrics['recall'] >= 0.9 else 'warning-metric' if metrics['recall'] >= 0.7 else 'error-metric'}">
                <h4>Recall</h4>
                <h2 style="color: {rec_color};">{metrics['recall']:.2%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            f1_color = get_metric_color(metrics['f1_score'], 'f1')
            st.markdown(f"""
            <div class="metric-card {'success-metric' if metrics['f1_score'] >= 0.9 else 'warning-metric' if metrics['f1_score'] >= 0.7 else 'error-metric'}">
                <h4>F1 Score</h4>
                <h2 style="color: {f1_color};">{metrics['f1_score']:.2%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ================================================================
        # CONFUSION MATRIX & ROC CURVE
        # ================================================================
        st.header("📈 Performance Visualizations")
        
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.subheader("Confusion Matrix")
            labels = sorted(np.unique(y_true))
            cm_fig = create_confusion_matrix_heatmap(
                metrics['confusion_matrix'], 
                labels, 
                normalize=show_normalized_cm
            )
            st.plotly_chart(cm_fig, use_container_width=True)
            
            # Confusion Matrix Statistics
            with st.expander("📊 Confusion Matrix Details"):
                cm_df = pd.DataFrame(
                    metrics['confusion_matrix'],
                    index=[f'Actual: {l}' for l in labels],
                    columns=[f'Predicted: {l}' for l in labels]
                )
                st.dataframe(cm_df, use_container_width=True)
        
        with col_viz2:
            if 'roc_data' in metrics:
                st.subheader("ROC Curve")
                roc_fig = create_roc_curve(
                    metrics['roc_data']['fpr'],
                    metrics['roc_data']['tpr'],
                    metrics['roc_data']['auc']
                )
                st.plotly_chart(roc_fig, use_container_width=True)
                
                st.metric(
                    "AUC Score", 
                    f"{metrics['roc_data']['auc']:.4f}",
                    delta=f"{(metrics['roc_data']['auc'] - 0.5):.4f} vs random",
                    delta_color="normal"
                )
            else:
                st.info("ROC Curve is only available for binary classification with probability scores.")
        
        # ================================================================
        # PER-CLASS METRICS
        # ================================================================
        if show_per_class:
            st.header("📋 Per-Class Performance Metrics")

            # Attempt to extract model training time if available
            training_time_display = 'N/A'
            if selected_model and selected_model.get('path') and os.path.exists(selected_model.get('path')):
                try:
                    loaded_meta = joblib.load(selected_model.get('path'))
                    # If a dict with metadata was saved
                    if isinstance(loaded_meta, dict) and 'training_time' in loaded_meta:
                        training_time_val = loaded_meta.get('training_time')
                    else:
                        # look for common attributes on estimator
                        training_time_val = None
                        for attr in ('training_time', 'fit_time', 'training_duration', 'time_elapsed'):
                            if hasattr(loaded_meta, attr):
                                training_time_val = getattr(loaded_meta, attr)
                                break
                    if training_time_val is not None:
                        try:
                            training_time_display = f"{float(training_time_val):.2f} s"
                        except Exception:
                            training_time_display = str(training_time_val)
                except Exception:
                    training_time_display = 'N/A'

            # Extract per-class metrics and compute per-class accuracy and ROC AUC when possible
            per_class_data = []
            cm = metrics.get('confusion_matrix')
            total = cm.sum() if cm is not None else 0

            # prepare probability matrix if available
            prob_cols = [col for col in df.columns if col.startswith('prob_')]
            proba_matrix = df[prob_cols].values if prob_cols else None

            from sklearn.metrics import roc_auc_score

            for idx, label in enumerate(labels):
                label_str = str(label)
                class_metrics = metrics['per_class_metrics'].get(label_str, {}) if 'per_class_metrics' in metrics else {}

                TP = int(cm[idx, idx]) if cm is not None else 0
                row_sum = int(cm[idx, :].sum()) if cm is not None else 0
                col_sum = int(cm[:, idx].sum()) if cm is not None else 0
                TN = int(total - row_sum - col_sum + TP) if total else 0
                accuracy_per_class = (TP + TN) / total if total else 0.0

                # ROC AUC per class (one-vs-rest) if probabilities exist
                roc_auc_val = None
                if proba_matrix is not None and proba_matrix.shape[1] > idx:
                    try:
                        y_true_binary = (df['actual'] == label).astype(int)
                        roc_auc_val = float(roc_auc_score(y_true_binary, proba_matrix[:, idx]))
                    except Exception:
                        roc_auc_val = None

                per_class_data.append({
                    'Class': label,
                    'Accuracy': accuracy_per_class,
                    'Precision': class_metrics.get('precision', 0),
                    'Recall': class_metrics.get('recall', 0),
                    'F1-Score': class_metrics.get('f1-score', 0),
                    'ROC AUC': roc_auc_val,
                    'Support': class_metrics.get('support', 0),
                    'Training Time': training_time_display
                })

            per_class_df = pd.DataFrame(per_class_data)

            # Display table with formatting
            fmt = {
                'Accuracy': '{:.2%}',
                'Precision': '{:.2%}',
                'Recall': '{:.2%}',
                'F1-Score': '{:.2%}',
                'ROC AUC': '{:.4f}',
                'Support': '{:.0f}'
            }

            # Replace None with NaN for formatting
            per_class_df['ROC AUC'] = per_class_df['ROC AUC'].apply(lambda x: np.nan if x is None else x)

            st.dataframe(
                per_class_df.style.format(fmt).background_gradient(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], cmap='RdYlGn'),
                use_container_width=True
            )

            # Per-class visualization (Precision/Recall/F1)
            fig = go.Figure()
            for metric in ['Precision', 'Recall', 'F1-Score']:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=per_class_df['Class'],
                    y=per_class_df[metric],
                    text=per_class_df[metric].apply(lambda x: f'{x:.2%}'),
                    textposition='auto'
                ))

            fig.update_layout(
                title='Per-Class Metrics Comparison',
                xaxis_title='Class',
                yaxis_title='Score',
                barmode='group',
                height=400,
                yaxis=dict(tickformat='.0%')
            )

            st.plotly_chart(fig, use_container_width=True)
        
        # ================================================================
        # PREDICTION ANALYSIS TABLE
        # ================================================================
        st.header("🔍 Detailed Prediction Analysis")
        
        # Add correctness indicator
        df['is_correct'] = df['actual'] == df['predicted']
        df['confidence'] = df[prob_cols].max(axis=1) if prob_cols else None
        
        # Filter controls
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            prediction_filter = st.selectbox(
                "Filter by Result",
                ["All", "Correct Only", "Incorrect Only"]
            )
        
        with col_filter2:
            class_filter = st.multiselect(
                "Filter by Actual Class",
                options=sorted(df['actual'].unique()),
                default=None
            )
        
        with col_filter3:
            if df['confidence'] is not None:
                conf_threshold = st.slider(
                    "Min Confidence Score",
                    0.0, 1.0, 0.0, 0.05
                )
        
        # Apply filters
        filtered_df = df.copy()
        
        if prediction_filter == "Correct Only":
            filtered_df = filtered_df[filtered_df['is_correct'] == True]
        elif prediction_filter == "Incorrect Only":
            filtered_df = filtered_df[filtered_df['is_correct'] == False]
        
        if class_filter:
            filtered_df = filtered_df[filtered_df['actual'].isin(class_filter)]
        
        if df['confidence'] is not None:
            filtered_df = filtered_df[filtered_df['confidence'] >= conf_threshold]
        
        st.info(f"Showing {len(filtered_df)} of {len(df)} predictions")
        
        # Display predictions with pagination
        total_pages = (len(filtered_df) - 1) // results_per_page + 1
        page = st.number_input('Page', min_value=1, max_value=max(1, total_pages), value=1)
        
        start_idx = (page - 1) * results_per_page
        end_idx = start_idx + results_per_page
        
        display_df = filtered_df.iloc[start_idx:end_idx].copy()
        
        # Color code the rows
        def highlight_prediction(row):
            if row['is_correct']:
                return ['background-color: #d1fae5'] * len(row)  # Light green
            else:
                return ['background-color: #fee2e2'] * len(row)  # Light red
        
        styled_df = display_df.style.apply(highlight_prediction, axis=1)
        
        if df['confidence'] is not None:
            styled_df = styled_df.format({'confidence': '{:.2%}'})
        
        st.dataframe(styled_df, use_container_width=True, height=600)
        
        # ================================================================
        # ERROR ANALYSIS
        # ================================================================
        st.header("🔬 Error Analysis")
        
        incorrect_df = df[df['is_correct'] == False].copy()
        
        if len(incorrect_df) > 0:
            col_err1, col_err2 = st.columns(2)
            
            with col_err1:
                st.subheader("Most Common Misclassifications")
                
                # Create misclassification pairs
                incorrect_df['misclassification'] = (
                    incorrect_df['actual'].astype(str) + ' → ' + 
                    incorrect_df['predicted'].astype(str)
                )
                
                misclass_counts = incorrect_df['misclassification'].value_counts().head(10)
                
                fig = go.Figure(go.Bar(
                    x=misclass_counts.values,
                    y=misclass_counts.index,
                    orientation='h',
                    marker=dict(color='#ef4444'),
                    text=misclass_counts.values,
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title='Top 10 Misclassification Patterns',
                    xaxis_title='Count',
                    yaxis_title='Actual → Predicted',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col_err2:
                st.subheader("Error Statistics")
                
                total_errors = len(incorrect_df)
                error_rate = (total_errors / len(df)) * 100
                
                st.metric("Total Errors", total_errors)
                st.metric("Error Rate", f"{error_rate:.2f}%")
                
                # Confidence score distribution for errors
                if df['confidence'] is not None:
                    st.write("**Confidence Distribution (Errors Only)**")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=incorrect_df['confidence'],
                        nbinsx=20,
                        marker=dict(color='#ef4444'),
                        name='Error Confidence'
                    ))
                    
                    fig.update_layout(
                        xaxis_title='Confidence Score',
                        yaxis_title='Count',
                        height=300,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Worst predictions showcase
                if df['confidence'] is not None:
                    st.write("**Worst Predictions (Lowest Confidence Errors)**")
                    worst_preds = incorrect_df.nsmallest(5, 'confidence')[
                        ['actual', 'predicted', 'confidence']
                    ]
                    st.dataframe(
                        worst_preds.style.format({'confidence': '{:.2%}'}),
                        use_container_width=True
                    )
        else:
            st.success("🎉 Perfect predictions! No errors found.")
        
        # ================================================================
        # EXPORT SECTION
        # ================================================================
        st.header("📥 Export Results")
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            # Export predictions to CSV
            csv_data = export_to_csv(df)
            st.download_button(
                label="📄 Download Predictions (CSV)",
                data=csv_data,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col_exp2:
            # Export metrics to JSON
            export_metrics = {
                'model_info': selected_model,
                'evaluation_date': datetime.now().isoformat(),
                'dataset_size': len(df),
                'metrics': {
                    'accuracy': float(metrics['accuracy']),
                    'precision': float(metrics['precision']),
                    'recall': float(metrics['recall']),
                    'f1_score': float(metrics['f1_score'])
                },
                'confusion_matrix': metrics['confusion_matrix'].tolist(),
                'per_class_metrics': metrics['per_class_metrics']
            }
            
            if 'roc_data' in metrics:
                export_metrics['auc'] = float(metrics['roc_data']['auc'])
            
            json_data = export_to_json(export_metrics)
            st.download_button(
                label="📊 Download Metrics (JSON)",
                data=json_data,
                file_name=f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col_exp3:
            st.info("📝 PDF Report generation will be available in the next update")
        
    except Exception as e:
        st.error(f"❌ Error processing dataset: {str(e)}")
        
        with st.expander("🔧 Error Details"):
            import traceback
            st.code(traceback.format_exc())

else:
    # ================================================================
    # LANDING PAGE - NO FILE UPLOADED
    # ================================================================
    st.info("📤 Please upload your test dataset to begin model evaluation")
    
    col_guide1, col_guide2 = st.columns(2)
    
    with col_guide1:
        st.subheader("📋 Required Dataset Format")
        st.markdown("""
        Your dataset should contain at minimum:
        
        - **`actual`**: True labels/ground truth
        - **`predicted`**: Model predictions
        - **`prob_class_0`, `prob_class_1`, ...** (optional): Probability scores for each class
        - Additional feature columns (optional)
        
        **Supported formats:** CSV, JSON, Excel (.xlsx)
        """)
        
        st.code("""
# Example CSV format:
actual,predicted,prob_0,prob_1,prob_2,feature_1,feature_2
0,0,0.92,0.05,0.03,5.1,3.5
1,1,0.03,0.94,0.03,4.9,3.0
2,2,0.02,0.03,0.95,6.3,2.9
        """, language="csv")
    
    with col_guide2:
        st.subheader("✨ Dashboard Features")
        st.markdown("""
        This dashboard provides:
        
        ✅ **Comprehensive Metrics**
        - Accuracy, Precision, Recall, F1-Score
        - Per-class performance breakdown
        
        ✅ **Visualizations**
        - Interactive confusion matrix
        - ROC curves with AUC scores
        - Performance comparisons
        
        ✅ **Detailed Analysis**
        - Prediction-level inspection
        - Error pattern analysis
        - Confidence score distributions
        
        ✅ **Export Options**
        - CSV predictions export
        - JSON metrics export
        - PDF report generation (coming soon)
        """)
    
    # Sample dataset generator
    st.subheader("🎲 Generate Sample Dataset")
    st.markdown("Don't have a dataset? Generate a sample for testing:")
    
    col_sample1, col_sample2, col_sample3 = st.columns(3)
    
    with col_sample1:
        n_samples = st.number_input("Number of samples", 100, 10000, 1000)
    with col_sample2:
        n_classes = st.number_input("Number of classes", 2, 10, 3)
    with col_sample3:
        accuracy_sim = st.slider("Simulated accuracy", 0.5, 1.0, 0.85, 0.05)
    
    if st.button("Generate Sample Dataset"):
        # Generate sample data
        np.random.seed(42)
        
        # Generate true labels
        y_true = np.random.randint(0, n_classes, n_samples)
        
        # Generate predictions based on accuracy
        y_pred = y_true.copy()
        n_errors = int(n_samples * (1 - accuracy_sim))
        error_indices = np.random.choice(n_samples, n_errors, replace=False)
        
        for idx in error_indices:
            # Change to a different class
            other_classes = [c for c in range(n_classes) if c != y_true[idx]]
            y_pred[idx] = np.random.choice(other_classes)
        
        # Generate probability scores
        prob_data = np.random.dirichlet(np.ones(n_classes) * 2, n_samples)
        
        # Adjust probabilities to match predictions
        for i in range(n_samples):
            prob_data[i, y_pred[i]] = np.random.uniform(0.7, 0.99)
            prob_data[i] = prob_data[i] / prob_data[i].sum()
        
        # Create DataFrame
        sample_df = pd.DataFrame({
            'actual': y_true,
            'predicted': y_pred
        })
        
        # Add probability columns
        for c in range(n_classes):
            sample_df[f'prob_{c}'] = prob_data[:, c]
        
        # Add some feature columns
        sample_df['feature_1'] = np.random.randn(n_samples)
        sample_df['feature_2'] = np.random.randn(n_samples)
        
        # Provide download
        csv_sample = export_to_csv(sample_df)
        st.success(f"✅ Generated {n_samples} samples with ~{accuracy_sim:.0%} accuracy")
        
        st.download_button(
            label="📥 Download Sample Dataset",
            data=csv_sample,
            file_name="sample_evaluation_dataset.csv",
            mime="text/csv"
        )
        
        st.dataframe(sample_df.head(20), use_container_width=True)

# ================================================================
# FOOTER
# ================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 20px;'>
    <p>AI Model Evaluation Dashboard | Built with Streamlit</p>
    <p>For questions or support, contact your ML team</p>
</div>
""", unsafe_allow_html=True)