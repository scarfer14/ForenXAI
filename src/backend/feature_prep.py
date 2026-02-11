import os
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Paths
# ===============================
# Get the absolute path to ensure paths work regardless of cwd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "feature_engineering")
ENCODER_DIR = os.path.join(BASE_DIR, "encoders")
SCALER_DIR = os.path.join(BASE_DIR, "scalers")

# For processed data, use project root
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
PROCESSED_DIR = os.path.abspath(os.path.join(ROOT_DIR, "data", "processed"))

os.makedirs(ENCODER_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Debug: Show where files will be saved
print(f"[feature_prep.py] Directories created:")
print(f"  Encoders: {ENCODER_DIR}")
print(f"  Scalers: {SCALER_DIR}")
print(f"  Processed: {PROCESSED_DIR}")
print()

# ===============================
# Helper Functions
# ===============================
def _combine_datasets(df_real: pd.DataFrame, df_synthetic: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Combine real and synthetic datasets."""
    if df_synthetic is not None:
        return pd.concat([df_real, df_synthetic], axis=0).reset_index(drop=True)
    return df_real.copy()

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features."""
    if 'feature1' in df.columns and 'feature2' in df.columns:
        df['ratio_feature'] = df['feature1'] / (df['feature2'] + 1e-5)
    
    if 'bytes_in' in df.columns and 'bytes_out' in df.columns:
        df['total_bytes'] = df['bytes_in'] + df['bytes_out']
    
    if 'timestamp' in df.columns:
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    
    return df

def _encode_categorical_fit(df: pd.DataFrame, cat_cols: list) -> Dict:
    """Fit and encode categorical variables."""
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    joblib.dump(encoders, os.path.join(ENCODER_DIR, "label_encoders.joblib"))
    return encoders

def _encode_categorical_transform(df: pd.DataFrame, cat_cols: list, encoders: Dict) -> None:
    """Transform categorical variables using fitted encoders."""
    for col in cat_cols:
        if col in encoders:
            df[col] = encoders[col].transform(df[col].astype(str))

def _fit_scalers(df: pd.DataFrame, num_cols: list) -> Tuple[PowerTransformer, StandardScaler]:
    """Fit and apply power transformer and scaler."""
    # Handle inf and NaN values before scaling
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df[num_cols] = df[num_cols].fillna(0)
    
    power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    df[num_cols] = power_transformer.fit_transform(df[num_cols])
    
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    joblib.dump(power_transformer, os.path.join(SCALER_DIR, "power_transformer.joblib"))
    joblib.dump(scaler, os.path.join(SCALER_DIR, "scaler.joblib"))
    
    return power_transformer, scaler

def _apply_scalers(df: pd.DataFrame, num_cols: list, 
                   power_transformer: PowerTransformer, scaler: StandardScaler) -> None:
    """Apply fitted scalers to data."""
    # Handle inf and NaN values before scaling
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df[num_cols] = df[num_cols].fillna(0)
    
    df[num_cols] = power_transformer.transform(df[num_cols])  # type: ignore[union-attr]
    df[num_cols] = scaler.transform(df[num_cols])  # type: ignore[union-attr]

# ===============================
# MAIN FEATURE PIPELINE
# ===============================
def prepare_features(
    df_real: pd.DataFrame, 
    df_synthetic: Optional[pd.DataFrame] = None, 
    fit: bool = True, 
    encoders: Optional[Dict] = None, 
    scaler: Optional[StandardScaler] = None, 
    power_transformer: Optional[PowerTransformer] = None
) -> Tuple[Dict[str, pd.DataFrame], Optional[Dict[str, Any]]]:
    """
    Returns:
        {
            'tree_models': unscaled features,
            'deep_learning': scaled features
        }
    """
    # 1. Combine datasets
    print("   Combining datasets...")
    df = _combine_datasets(df_real, df_synthetic)
    print(f"   ✅ Combined: {df.shape}")
    
    # 2. Feature Engineering
    print("⏳ Step 2/5: Engineering features...")
    df = _engineer_features(df)
    
    # 3. Encode Categorical Variables
    print("⏳ Step 3/5: Encoding categorical variables...")
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if fit:
        encoders = _encode_categorical_fit(df, cat_cols)
    else:
        if encoders is None:
            raise ValueError("encoders must be provided when fit=False")
        _encode_categorical_transform(df, cat_cols, encoders)
    
    # 4. Branch for model types
    df_trees = df.copy()  # Unscaled for RF / Isolation Forest
    
    # 5. Transform + Scale (Deep Learning only)
    print("⏳ Step 4/5: Power transformation (this takes time)...")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if fit:
        power_transformer, scaler = _fit_scalers(df, num_cols)
    else:
        if power_transformer is None or scaler is None:
            raise ValueError("power_transformer and scaler must be provided when fit=False")
        _apply_scalers(df, num_cols, power_transformer, scaler)
    
    # 6. Save processed training data
    print("⏳ Step 5/5: Saving processed data...")
    if fit:
        df_trees.to_csv(os.path.join(PROCESSED_DIR, "train_features_trees.csv"), index=False)
        df.to_csv(os.path.join(PROCESSED_DIR, "train_features_dl.csv"), index=False)
    
    outputs = {
        "tree_models": df_trees,
        "deep_learning": df
    }
    
    artifacts = {
        "encoders": encoders,
        "scaler": scaler,
        "power_transformer": power_transformer
    } if fit else None
    
    return outputs, artifacts


# ===============================
# TF-IDF TEXT FEATURES (Supervised ML)
# ===============================
def prepare_tfidf(df, text_col='text_feature', fit=True, tfidf_vectorizer=None):
    if fit:
        tfidf = TfidfVectorizer(max_features=1000)
        tfidf_matrix = tfidf.fit_transform(df[text_col])
        joblib.dump(tfidf, os.path.join(ENCODER_DIR, "tfidf_vectorizer.joblib"))

        # Save vocabulary for SHAP alignment
        vocab = tfidf.get_feature_names_out()
        joblib.dump(vocab, os.path.join(ENCODER_DIR, "tfidf_vocab.joblib"))
    else:
        if tfidf_vectorizer is None:
            raise ValueError("tfidf_vectorizer must be provided when fit=False")
        tfidf = tfidf_vectorizer
        tfidf_matrix = tfidf.transform(df[text_col])  # type: ignore[union-attr]

    return tfidf_matrix


# ===============================
# Isolation Forest Feature Frame
# ===============================
def save_iso_features(df, features):
    iso_df = df[features].copy()
    iso_df.fillna(0, inplace=True)
    joblib.dump(features, os.path.join(ENCODER_DIR, "iso_features_list.joblib"))
    return iso_df


# ===============================
# Visualization (Optional)
# ===============================
def visualize_features(df, numeric_cols=None):
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    for col in numeric_cols:
        plt.figure(figsize=(6, 3))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.show()
