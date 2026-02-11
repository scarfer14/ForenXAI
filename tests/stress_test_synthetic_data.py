"""
Stress Testing with Real Synthetic Dataset
==========================================
Professional stress testing using the actual synthetic network traffic data
from data/synthetic/ directory. Tests model performance with realistic
attack patterns and distributions.

Features:
- Uses real synthetic data (44K training samples, validation samples)
- Tests all three models (Random Forest, MLP, Isolation Forest)
- Measures throughput, latency, accuracy, and memory usage
- Compares predictions against ground truth labels
- Generates detailed performance reports

Author: ForenXAI Team
Date: February 2026
"""

import numpy as np
import pandas as pd
import joblib
import time
import psutil
import os
from pathlib import Path
from tensorflow.keras.models import load_model
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Determine paths based on environment
IN_COLAB = False
try:
    from google.colab import drive
    IN_COLAB = True
    DATA_DIR = '/content/drive/MyDrive/ForenXAI/data/synthetic'
    MODELS_DIR = '/content/drive/MyDrive/Featured Dataset/trained_models'
except:
    SCRIPT_DIR = Path(__file__).parent.absolute()
    DATA_DIR = SCRIPT_DIR.parent / 'data' / 'synthetic'
    MODELS_DIR = SCRIPT_DIR.parent / 'models'

# Data files
TRAIN_FILE = 'synthetic_train_split.csv'
VAL_FILE = 'synthetic_val_split.csv'

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_memory_usage():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_cpu_usage():
    """Get current CPU usage percentage"""
    return psutil.cpu_percent(interval=0.1)

def load_synthetic_data(file_path, sample_size=None):
    """
    Load synthetic data from CSV
    
    Parameters:
    - file_path: Path to CSV file
    - sample_size: Optional number of samples to load (for stress levels)
    
    Returns:
    - X: Features array
    - y: Labels array (0=Normal/Benign, 1=Attack)
    - attack_types: Attack type labels
    """
    print(f"  Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)
        print(f"  Sampled {sample_size:,} rows from {len(df):,} total")
    
    # Extract features (exclude Label and Attack columns)
    X = df.drop(columns=['Label', 'Attack'], errors='ignore').values
    
    # Extract labels (convert to binary: 0=Benign, 1=Attack)
    if 'Label' in df.columns:
        y = df['Label'].values
    else:
        # If no Label, create from Attack (Benign=0, anything else=1)
        y = (df['Attack'] != 'Benign').astype(int).values
    
    attack_types = df['Attack'].values if 'Attack' in df.columns else None
    
    print(f"  Loaded {len(X):,} samples with {X.shape[1]} features")
    print(f"  Label distribution: Normal={np.sum(y==0):,} ({np.mean(y==0)*100:.1f}%), "
          f"Attack={np.sum(y==1):,} ({np.mean(y==1)*100:.1f}%)")
    
    return X, y, attack_types

def preprocess_data(X):
    """Clean and preprocess data for model input"""
    # Handle NaN and Inf values
    X_clean = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    return X_clean

# ============================================================
# MODEL STRESS TESTERS
# ============================================================

class SyntheticDataStressTester:
    """Base class for stress testing with synthetic data"""
    
    def __init__(self, model_path, model_name):
        self.model_path = model_path
        self.model_name = model_name
        self.model = None
        
    def load_model(self):
        """Load the trained model"""
        raise NotImplementedError
        
    def predict(self, X):
        """Make predictions"""
        raise NotImplementedError
        
    def run_performance_test(self, X, y, attack_types=None):
        """
        Run comprehensive performance test with ground truth
        """
        print(f"\n{'='*70}")
        print(f"PERFORMANCE TEST - {self.model_name}")
        print(f"{'='*70}")
        
        n_samples = len(X)
        mem_before = get_memory_usage()
        cpu_before = get_cpu_usage()
        
        # Prediction with timing
        print(f"\n  Processing {n_samples:,} predictions...")
        start_time = time.time()
        predictions = self.predict(X)
        elapsed_time = time.time() - start_time
        
        mem_after = get_memory_usage()
        cpu_after = get_cpu_usage()
        
        # Calculate performance metrics
        throughput = n_samples / elapsed_time
        latency_per_sample = (elapsed_time / n_samples) * 1000  # ms
        
        # Calculate accuracy metrics
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, zero_division=0)
        recall = recall_score(y, predictions, zero_division=0)
        f1 = f1_score(y, predictions, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # Results
        results = {
            'n_samples': n_samples,
            'elapsed_time_s': elapsed_time,
            'throughput_samples_per_sec': throughput,
            'latency_per_sample_ms': latency_per_sample,
            'memory_delta_mb': mem_after - mem_before,
            'cpu_avg': (cpu_before + cpu_after) / 2,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
        
        # Print results
        print(f"\n  ⚡ PERFORMANCE METRICS:")
        print(f"     Total Time: {elapsed_time:.3f} seconds")
        print(f"     Throughput: {throughput:,.0f} samples/sec")
        print(f"     Latency: {latency_per_sample:.4f} ms/sample")
        print(f"     Memory Delta: {mem_after - mem_before:.2f} MB")
        print(f"     CPU Usage: {(cpu_before + cpu_after)/2:.1f}%")
        
        print(f"\n  📊 ACCURACY METRICS:")
        print(f"     Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"     Precision: {precision:.4f}")
        print(f"     Recall:    {recall:.4f}")
        print(f"     F1-Score:  {f1:.4f}")
        
        print(f"\n  🎯 CONFUSION MATRIX:")
        print(f"     True Negatives:  {tn:,} (correctly identified normal)")
        print(f"     False Positives: {fp:,} (normal flagged as attack)")
        print(f"     False Negatives: {fn:,} (missed attacks)")
        print(f"     True Positives:  {tp:,} (correctly detected attacks)")
        
        # Attack-type breakdown if available
        if attack_types is not None:
            print(f"\n  🔍 ATTACK TYPE BREAKDOWN:")
            unique_attacks = np.unique(attack_types)
            for attack in unique_attacks:
                mask = attack_types == attack
                if np.sum(mask) > 0:
                    attack_acc = accuracy_score(y[mask], predictions[mask])
                    print(f"     {attack:<20}: {attack_acc:.4f} ({np.sum(mask):,} samples)")
        
        return results
    
    def run_stress_levels_test(self, X_full, y_full, stress_levels):
        """Test performance across different data volumes"""
        print(f"\n{'='*70}")
        print(f"STRESS LEVELS TEST - {self.model_name}")
        print(f"{'='*70}")
        
        results = []
        
        for level_name, sample_size in stress_levels.items():
            if sample_size > len(X_full):
                print(f"\n  ⚠️  Skipping {level_name}: Requested {sample_size:,} but only {len(X_full):,} available")
                continue
            
            print(f"\n  [{level_name.upper()}] Testing with {sample_size:,} samples...")
            
            # Sample data
            indices = np.random.choice(len(X_full), sample_size, replace=False)
            X_sample = X_full[indices]
            y_sample = y_full[indices]
            
            # Run test
            start = time.time()
            predictions = self.predict(X_sample)
            elapsed = time.time() - start
            
            accuracy = accuracy_score(y_sample, predictions)
            throughput = sample_size / elapsed
            
            results.append({
                'level': level_name,
                'samples': sample_size,
                'time_s': elapsed,
                'throughput': throughput,
                'accuracy': accuracy
            })
            
            print(f"     Time: {elapsed:.3f}s | Throughput: {throughput:,.0f} s/s | Accuracy: {accuracy:.4f}")
        
        return results


class RandomForestSyntheticTester(SyntheticDataStressTester):
    """Stress tester for Random Forest with synthetic data"""
    
    def load_model(self):
        print(f"  📦 Loading Random Forest model...")
        self.model = joblib.load(self.model_path)
        print(f"  ✅ Model loaded successfully")
        
    def predict(self, X):
        X_clean = preprocess_data(X)
        return self.model.predict(X_clean)


class MLPSyntheticTester(SyntheticDataStressTester):
    """Stress tester for MLP with synthetic data"""
    
    def load_model(self):
        print(f"  📦 Loading MLP Neural Network model...")
        self.model = load_model(self.model_path, compile=False)
        print(f"  ✅ Model loaded successfully")
        
    def predict(self, X):
        X_clean = preprocess_data(X)
        predictions = self.model.predict(X_clean, verbose=0).flatten()
        return (predictions > 0.5).astype(int)


class IsolationForestSyntheticTester(SyntheticDataStressTester):
    """Stress tester for Isolation Forest with synthetic data"""
    
    def load_model(self):
        print(f"  📦 Loading Isolation Forest model...")
        self.model = joblib.load(self.model_path)
        print(f"  ✅ Model loaded successfully")
        
    def predict(self, X):
        X_clean = preprocess_data(X)
        predictions = self.model.predict(X_clean)
        return np.where(predictions == -1, 1, 0)


# ============================================================
# MAIN STRESS TEST SUITE
# ============================================================

def run_synthetic_data_stress_tests(use_validation=False, sample_size=None):
    """
    Run comprehensive stress tests using real synthetic data
    
    Parameters:
    - use_validation: If True, use validation split; otherwise use training split
    - sample_size: Optional limit on number of samples to test
    """
    
    print("="*70)
    print("SYNTHETIC DATA STRESS TESTING SUITE")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"System Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"CPU Cores: {psutil.cpu_count()}")
    print("="*70)
    
    # Load synthetic data
    data_file = VAL_FILE if use_validation else TRAIN_FILE
    data_path = os.path.join(DATA_DIR, data_file)
    
    if not os.path.exists(data_path):
        print(f"\n❌ ERROR: Data file not found at {data_path}")
        return None
    
    print(f"\n📂 Loading synthetic data from: {data_file}")
    X, y, attack_types = load_synthetic_data(data_path, sample_size)
    
    # Define model configurations
    models_config = [
        {
            'name': 'Random Forest',
            'path': os.path.join(MODELS_DIR, 'random_forest_pipeline.joblib'),
            'tester_class': RandomForestSyntheticTester
        },
        {
            'name': 'MLP Neural Network',
            'path': os.path.join(MODELS_DIR, 'mlp_model.h5'),
            'tester_class': MLPSyntheticTester
        },
        {
            'name': 'Isolation Forest',
            'path': os.path.join(MODELS_DIR, 'isolation_forest_pipeline.joblib'),
            'tester_class': IsolationForestSyntheticTester
        }
    ]
    
    all_results = {}
    
    # Test each model
    for model_config in models_config:
        model_name = model_config['name']
        model_path = model_config['path']
        
        if not os.path.exists(model_path):
            print(f"\n⚠️  Skipping {model_name}: Model file not found")
            continue
        
        print(f"\n{'#'*70}")
        print(f"# TESTING: {model_name}")
        print(f"{'#'*70}")
        
        try:
            # Initialize tester
            tester = model_config['tester_class'](model_path, model_name)
            tester.load_model()
            
            # Run performance test
            results = tester.run_performance_test(X, y, attack_types)
            
            # Run stress levels test if we have enough data
            if len(X) >= 10000:
                stress_levels = {
                    'mini': 1_000,
                    'small': 5_000,
                    'medium': 10_000,
                    'large': min(20_000, len(X))
                }
                stress_results = tester.run_stress_levels_test(X, y, stress_levels)
                results['stress_levels'] = stress_results
            
            all_results[model_name] = results
            
        except Exception as e:
            print(f"\n❌ Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_name] = {'error': str(e)}
    
    # ============================================================
    # SUMMARY REPORT
    # ============================================================
    print("\n" + "="*70)
    print("STRESS TEST SUMMARY - SYNTHETIC DATA")
    print("="*70)
    print(f"Dataset: {data_file} ({len(X):,} samples)")
    print(f"Attack Types: {len(np.unique(attack_types))} unique types")
    print("="*70)
    
    # Performance comparison table
    print(f"\n{'Model':<25} {'Throughput':<20} {'Latency':<15} {'Accuracy':<12} {'F1-Score':<12}")
    print("-"*84)
    
    for model_name, results in all_results.items():
        if 'error' not in results:
            throughput = f"{results['throughput_samples_per_sec']:,.0f} s/s"
            latency = f"{results['latency_per_sample_ms']:.4f} ms"
            accuracy = f"{results['accuracy']:.4f}"
            f1 = f"{results['f1_score']:.4f}"
            
            print(f"{model_name:<25} {throughput:<20} {latency:<15} {accuracy:<12} {f1:<12}")
        else:
            print(f"{model_name:<25} {'ERROR':<20} {results['error'][:40]}")
    
    # Best performer
    print("\n" + "="*70)
    print("BEST PERFORMERS")
    print("="*70)
    
    valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
    
    if valid_results:
        best_accuracy = max(valid_results.items(), key=lambda x: x[1]['accuracy'])
        best_throughput = max(valid_results.items(), key=lambda x: x[1]['throughput_samples_per_sec'])
        best_f1 = max(valid_results.items(), key=lambda x: x[1]['f1_score'])
        
        print(f"🎯 Best Accuracy:  {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.4f})")
        print(f"⚡ Best Throughput: {best_throughput[0]} ({best_throughput[1]['throughput_samples_per_sec']:,.0f} s/s)")
        print(f"📊 Best F1-Score:  {best_f1[0]} ({best_f1[1]['f1_score']:.4f})")
    
    print("\n" + "="*70)
    print("✅ SYNTHETIC DATA STRESS TESTING COMPLETE")
    print("="*70)
    
    return all_results


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ForenXAI Synthetic Data Stress Testing')
    parser.add_argument('--validation', action='store_true',
                        help='Use validation split instead of training split')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Limit number of samples to test (default: use all)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to data directory (overrides default)')
    parser.add_argument('--models-dir', type=str, default=None,
                        help='Path to models directory (overrides default)')
    
    args = parser.parse_args()
    
    # Override directories if specified
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)
    if args.models_dir:
        MODELS_DIR = Path(args.models_dir)
    
    # Run stress tests
    results = run_synthetic_data_stress_tests(
        use_validation=args.validation,
        sample_size=args.sample_size
    )
    
    print(f"\n💾 Test complete! Results available in memory.")
    print(f"📊 Use --validation flag to test on validation split")
    print(f"⚡ Use --sample-size N to limit testing to N samples")
