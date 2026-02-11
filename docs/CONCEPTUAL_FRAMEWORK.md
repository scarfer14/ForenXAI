# Conceptual Framework: ForenXAI

## Input-Process-Output Model

**ForenXAI** is an AI-assisted cyber forensics system that transforms raw network flow data into actionable threat intelligence through explainable machine learning. The framework addresses the critical challenge of manual forensic analysis being too slow and labor-intensive for modern cyber threats by automating detection while maintaining human interpretability.

### **INPUTS**
Forensic analysts upload CSV files containing network flow data (payload timelines) through the Streamlit dashboard interface. These files include fields such as payload content, protocol types, source/destination IPs, timestamps, and connection metadata from networks under investigation. The system accepts both real-time captured traffic and historical network logs that require forensic analysis. The uploaded data represents potentially malicious network activity that needs automated classification and explanation.

*Note: During development, the system was trained on the NF-UNSW-NB15-v3 dataset (2.3M flows) augmented with Gretel ACTGAN synthetic samples to ensure robust attack pattern recognition.*

### **PROCESSES**
The system processes uploaded data through pre-trained AI models that were developed in a multi-phase pipeline:

**Feature Preparation:** User-uploaded CSV data is validated and cleaned. Text features are vectorized using TF-IDF (Term Frequency-Inverse Document Frequency), converting payload content into numeric representations. Numeric features are extracted from timestamps (hour of day) and IP addresses (last octet). These engineered features feed into the two prediction models.

**Model 1 - Severity Classification (Random Forest):** Uses TF-IDF text features to predict **threat severity levels** (benign, low, medium, high-risk). The Random Forest ensemble classifier was trained on labeled attack data to recognize textual patterns associated with different threat categories (DoS, reconnaissance, backdoors, etc.).

**Model 2 - Anomaly Detection (Isolation Forest):** Uses engineered numeric features to identify **anomalous network flows**. The Isolation Forest algorithm isolates outliers—flagging unusual temporal or IP-based patterns that deviate from normal behavior—without requiring labeled anomaly examples.

**Explainability:** An explainability engine shows analysts why the AI made each decision by highlighting which specific words, timestamps, or IP patterns most influenced the prediction, enabling transparent validation of AI reasoning.

### **OUTPUTS**
The system produces two key outputs delivered through an interactive Streamlit dashboard: (1) **Predictions** - multi-class severity scores and binary anomaly flags with confidence metrics, and (2) **Explanations** - visualizations showing which features (words, temporal patterns, IP characteristics) drove each prediction. These outputs enable forensic analysts to rapidly triage incidents, understand AI reasoning, and make informed decisions about threat prioritization.

### **THEORETICAL RELATIONSHIP**
The framework establishes a causal chain: diverse network inputs → automated feature extraction and dual-model processing → explainable predictions. This relationship transforms unstructured forensic data into interpretable intelligence, reducing analysis time through automated detection while maintaining forensic validity through transparent AI explanations that show which specific features influenced each decision.

---

## System Architecture

ForenXAI employs a four-tier architecture that separates user interaction, application orchestration, machine learning processing, and infrastructure resources into distinct logical components, ensuring maintainability, scalability, and clear separation of concerns for cyber forensic analysis.

The presentation layer consists of a Streamlit web dashboard (`ai_dashboard.py`) where forensic analysts upload CSV files containing network flow data, view real-time threat predictions, and examine interactive explainability visualizations. The interface displays multi-class severity classifications, binary anomaly flags with confidence scores, and SHAP-based feature importance charts that highlight which specific payload words, timestamps, or IP patterns contributed to each prediction. Users access the dashboard through a localhost browser, enabling immediate feedback during forensic investigations.

The application layer manages data preprocessing workflows through request handlers and feature preparation pipelines (`frontend_ingest.py`, `feature_prep.py`). This tier validates CSV inputs against expected schemas, sanitizes network flow data by handling missing values and outliers, applies TF-IDF vectorization to payload text fields, and extracts engineered numeric features from timestamps (hour of day) and IP addresses (last octet). The layer transforms raw, heterogeneous CSV data into standardized ML-ready feature matrices that feed both prediction models simultaneously.

The domain logic layer contains two pre-trained models executing in parallel: a Random Forest classifier (`severity_clf.joblib`) for multi-class threat severity prediction and an Isolation Forest (`anomaly_iso.joblib`) for unsupervised anomaly detection. Streamlit loads these serialized models directly from disk and runs inference on engineered features without intermediate orchestration layers. An integrated SHAP explainability engine computes Shapley values post-prediction, decomposing model decisions into individual feature contributions for transparent AI reasoning. Future work includes integrating a deep learning model for performance benchmarking against the current Random Forest classifier.

The infrastructure layer encompasses both logical and physical resources. Logical components include serialized model files (`.joblib`), feature transformation artifacts (TF-IDF vocabularies, scalers), training datasets (NF-UNSW-NB15-v3 with 2.3M flows plus Gretel ACTGAN synthetic data), and logging services. Physical components include the Python 3.x runtime with core libraries (scikit-learn, pandas, numpy, SHAP, Streamlit), local file storage, and computational resources operating on a single localhost development server. The system requires a multi-core CPU (minimum 4 cores recommended for parallel Random Forest inference), at least 8GB RAM for loading models and processing network flow datasets, and approximately 5GB local disk storage for datasets, serialized models, and feature artifacts. No GPU is required as the current models use CPU-based scikit-learn algorithms.

### **Architecture Diagram** *(See Appendix A)*

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE LAYER                         │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │         Streamlit Web Dashboard (ai_dashboard.py)             │  │
│  │  • File Upload  • Prediction Display  • SHAP Visualizations   │  │
│  └────────────────────────────┬──────────────────────────────────┘  │
└────────────────────────────────┼─────────────────────────────────────┘
                                 │ HTTP/WebSocket
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      APPLICATION SERVER LAYER                        │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │    Data Ingestion & Feature Preparation                       │  │
│  │    (frontend_ingest.py, feature_prep.py)                      │  │
│  │  • CSV Validation  • Data Cleaning  • TF-IDF Vectorization    │  │
│  │  • Numeric Feature Extraction (Hour, IP Last Octet)           │  │
│  └────────────────────────────┬──────────────────────────────────┘  │
└────────────────────────────────┼─────────────────────────────────────┘
                                 │ Feature Vectors
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         ML ENGINE LAYER                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              Model Inference (model_inference.py)            │   │
│  │  ┌─────────────────────┐      ┌─────────────────────┐       │   │
│  │  │  Random Forest      │      │  Isolation Forest   │       │   │
│  │  │  Severity Classifier│      │  Anomaly Detector   │       │   │
│  │  │  (TF-IDF → RF)      │      │  (Numeric → ISO)    │       │   │
│  │  └─────────────────────┘      └─────────────────────┘       │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │     SHAP Explainability Engine                      │    │   │
│  │  │     (Feature Importance Computation)                │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  └────────────────────────────┬─────────────────────────────────┘   │
└────────────────────────────────┼─────────────────────────────────────┘
                                 │ Model Loading
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA STORAGE LAYER                              │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    Training Dataset                           │  │
│  │   • NF-UNSW-NB15-v3 (2.3M flows)                              │  │
│  │   • Gretel ACTGAN Synthetic Data (5K+ samples)                │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              Serialized Model Artifacts (models/)             │  │
│  │   • severity_clf.joblib (Random Forest Pipeline)              │  │
│  │   • anomaly_iso.joblib (Isolation Forest)                     │  │
│  │   • tfidf_vocab.pkl (TF-IDF Vectorizer)                       │  │
│  │   • shap_background.pkl (SHAP Explainer Data)                 │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │          Development Notebooks (src/backend/)                 │  │
│  │   • synthetic_data_generation.ipynb                           │  │
│  │   • data_prep.ipynb  • feature_prep.ipynb                     │  │
│  │   • train_model.ipynb  • model_evaluation.ipynb               │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

**Component Interactions:**
1. **User → UI Layer:** Analyst uploads CSV file via Streamlit web interface
2. **UI → Application Server:** File transferred to ingestion module for validation
3. **Application Server → ML Engine:** Feature vectors passed to inference module
4. **ML Engine ↔ Data Storage:** Models and artifacts loaded from disk
5. **ML Engine → Application Server:** Predictions + SHAP explanations returned
6. **Application Server → UI:** Results formatted and displayed in dashboard

---

## References

- **Machine Learning Pipeline**: [IBM Machine Learning Pipeline](https://www.ibm.com/think/topics/machine-learning-pipeline)
- **SHAP**: Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS.
- **Isolation Forest**: Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. ICDM.
- **ACTGAN**: Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). Modeling tabular data using conditional GAN. NeurIPS.
- **NF-UNSW-NB15-v3 Dataset**: [University of Queensland NIDS Datasets](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)

---

**Document Version**: 1.0  
**Last Updated**: February 4, 2026  
**Authors**: ForenXAI Development Team
