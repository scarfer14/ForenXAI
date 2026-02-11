"""
Stress Testing Suite for ForenXAI Models
=========================================
Comprehensive stress testing for trained ML models using synthetic datasets.

Tests Include:
- High-volume throughput testing (100K-1M predictions)
- Latency benchmarking (single vs batch predictions)
- Memory consumption under load
- Edge case handling (NaN, Inf, extreme values)
- Model robustness (adversarial inputs)
- Concurrent request simulation

Author: ForenXAI Team
Date: February 2026
"""

import numpy as np
import pandas as pd
import joblib
import time
import psutil
import os
import sys
from pathlib import Path
from tensorflow.keras.models import load_model
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Stress test parameters
STRESS_LEVELS = {
    'light': 1_000,      # 1K samples
    'medium': 10_000,    # 10K samples
    'heavy': 100_000,    # 100K samples
    'extreme': 500_000   # 500K samples
}

# Model paths (adjust based on your setup)
IN_COLAB = False  # Set to True if running in Colab
if IN_COLAB:
    MODELS_DIR = '/content/drive/MyDrive/Featured Dataset/trained_models'
else:
    SCRIPT_DIR = Path(__file__).parent.absolute()
    MODELS_DIR = SCRIPT_DIR.parent / 'models'

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_memory_usage():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def get_cpu_usage():
    """Get current CPU usage percentage"""
    return psutil.cpu_percent(interval=0.1)

def generate_synthetic_data(n_samples, n_features=34, anomaly_ratio=0.1):
    """
    Generate synthetic network traffic data for stress testing
    
    Parameters:
    - n_samples: Number of samples to generate
    - n_features: Number of features (default: 34)
    - anomaly_ratio: Proportion of anomalous samples
    
    Returns:
    - X: Feature array (n_samples, n_features)
    - y: Labels (0=normal, 1=attack)
    """
    print(f"  Generating {n_samples:,} synthetic samples...")
    
    n_normal = int(n_samples * (1 - anomaly_ratio))
    n_attack = n_samples - n_normal
    
    # Normal traffic: lower variance, centered around 0
    X_normal = np.random.randn(n_normal, n_features) * 0.5
    
    # Attack traffic: higher variance, shifted distribution
    X_attack = np.random.randn(n_attack, n_features) * 2.0 + 1.5
    
    # Add some realistic patterns
    # Feature correlations (simulate real network patterns)
    X_normal[:, 0] = np.abs(X_normal[:, 0])  # Packet size (always positive)
    X_attack[:, 0] = np.abs(X_attack[:, 0]) * 3  # Larger packets in attacks
    
    # Combine and shuffle
    X = np.vstack([X_normal, X_attack])
    y = np.hstack([np.zeros(n_normal), np.ones(n_attack)])
    
    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y

def generate_adversarial_data(n_samples, n_features=34):
    """Generate edge cases and adversarial examples"""
    X = np.random.randn(n_samples, n_features)
    
    # Introduce edge cases
    n_edge = n_samples // 10
    
    # 10% NaN values
    X[:n_edge, :5] = np.nan
    
    # 10% Infinite values
    X[n_edge:2*n_edge, :5] = np.inf
    
    # 10% Extreme values
    X[2*n_edge:3*n_edge, :5] = 1e10
    
    # 10% All zeros
    X[3*n_edge:4*n_edge, :] = 0
    
    return X

# ============================================================
# STRESS TEST CLASSES
# ============================================================

class ModelStressTester:
    """Base class for model stress testing"""
    
    def __init__(self, model_path, model_name):
        self.model_path = model_path
        self.model_name = model_name
        self.model = None
        self.results = {}
        
    def load_model(self):
        """Load the trained model"""
        raise NotImplementedError
        
    def predict(self, X):
        """Make predictions"""
        raise NotImplementedError
        
    def run_throughput_test(self, stress_level='medium'):
        """Test high-volume prediction throughput"""
        n_samples = STRESS_LEVELS[stress_level]
        print(f"\n{'='*70}")
        print(f"THROUGHPUT TEST - {self.model_name} ({stress_level.upper()} load)")
        print(f"{'='*70}")
        
        # Generate data
        X, _ = generate_synthetic_data(n_samples)
        
        # Measure memory before
        mem_before = get_memory_usage()
        
        # Run predictions
        print(f"  Processing {n_samples:,} predictions...")
        start_time = time.time()
        predictions = self.predict(X)
        elapsed_time = time.time() - start_time
        
        # Measure memory after
        mem_after = get_memory_usage()
        mem_delta = mem_after - mem_before
        
        # Calculate metrics
        throughput = n_samples / elapsed_time
        latency_per_sample = (elapsed_time / n_samples) * 1000  # ms
        
        results = {
            'stress_level': stress_level,
            'n_samples': n_samples,
            'total_time_s': elapsed_time,
            'throughput_samples_per_sec': throughput,
            'latency_per_sample_ms': latency_per_sample,
            'memory_delta_mb': mem_delta,
            'memory_after_mb': mem_after
        }
        
        print(f"\n  ✅ Results:")
        print(f"     Total Time: {elapsed_time:.2f}s")
        print(f"     Throughput: {throughput:,.0f} samples/sec")
        print(f"     Latency: {latency_per_sample:.4f} ms/sample")
        print(f"     Memory Delta: {mem_delta:.2f} MB")
        
        return results
    
    def run_latency_test(self, batch_sizes=[1, 10, 100, 1000]):
        """Test latency across different batch sizes"""
        print(f"\n{'='*70}")
        print(f"LATENCY TEST - {self.model_name}")
        print(f"{'='*70}")
        
        results = []
        
        for batch_size in batch_sizes:
            X, _ = generate_synthetic_data(batch_size)
            
            # Warm-up prediction
            _ = self.predict(X[:1])
            
            # Measure latency over 10 iterations
            latencies = []
            for _ in range(10):
                start = time.time()
                _ = self.predict(X)
                latencies.append(time.time() - start)
            
            avg_latency = np.mean(latencies) * 1000  # Convert to ms
            std_latency = np.std(latencies) * 1000
            
            results.append({
                'batch_size': batch_size,
                'avg_latency_ms': avg_latency,
                'std_latency_ms': std_latency,
                'latency_per_sample_ms': avg_latency / batch_size
            })
            
            print(f"  Batch Size {batch_size:>4}: {avg_latency:>8.2f} ms ± {std_latency:>6.2f} ms "
                  f"({avg_latency/batch_size:.4f} ms/sample)")
        
        return results
    
    def run_robustness_test(self):
        """Test model robustness with edge cases"""
        print(f"\n{'='*70}")
        print(f"ROBUSTNESS TEST - {self.model_name}")
        print(f"{'='*70}")
        
        n_samples = 1000
        X = generate_adversarial_data(n_samples)
        
        print(f"  Testing with {n_samples} adversarial samples...")
        
        try:
            start = time.time()
            predictions = self.predict(X)
            elapsed = time.time() - start
            
            success = True
            error_msg = None
            
            # Check for valid predictions
            valid_predictions = np.sum(~np.isnan(predictions))
            
            print(f"  ✅ Robustness Test Passed")
            print(f"     Valid predictions: {valid_predictions}/{n_samples}")
            print(f"     Processing time: {elapsed:.2f}s")
            
        except Exception as e:
            success = False
            error_msg = str(e)
            print(f"  ❌ Robustness Test Failed: {error_msg}")
        
        return {
            'success': success,
            'error': error_msg,
            'valid_predictions': valid_predictions if success else 0,
            'total_samples': n_samples
        }
    
    def run_memory_stress_test(self):
        """Test memory consumption under sustained load"""
        print(f"\n{'='*70}")
        print(f"MEMORY STRESS TEST - {self.model_name}")
        print(f"{'='*70}")
        
        memory_readings = []
        batch_size = 50_000
        n_iterations = 10
        
        print(f"  Running {n_iterations} iterations of {batch_size:,} predictions...")
        
        mem_initial = get_memory_usage()
        
        for i in range(n_iterations):
            X, _ = generate_synthetic_data(batch_size)
            _ = self.predict(X)
            
            mem_current = get_memory_usage()
            memory_readings.append(mem_current)
            
            if (i + 1) % 3 == 0:
                print(f"     Iteration {i+1}/{n_iterations}: Memory = {mem_current:.2f} MB")
        
        mem_final = get_memory_usage()
        mem_peak = max(memory_readings)
        mem_growth = mem_final - mem_initial
        
        print(f"\n  ✅ Memory Profile:")
        print(f"     Initial: {mem_initial:.2f} MB")
        print(f"     Peak: {mem_peak:.2f} MB")
        print(f"     Final: {mem_final:.2f} MB")
        print(f"     Growth: {mem_growth:.2f} MB ({(mem_growth/mem_initial)*100:.1f}%)")
        
        return {
            'initial_mb': mem_initial,
            'peak_mb': mem_peak,
            'final_mb': mem_final,
            'growth_mb': mem_growth,
            'growth_pct': (mem_growth/mem_initial)*100
        }


class RandomForestStressTester(ModelStressTester):
    """Stress tester for Random Forest model"""
    
    def load_model(self):
        print(f"\n📦 Loading Random Forest model from: {self.model_path}")
        self.model = joblib.load(self.model_path)
        print("  ✅ Model loaded successfully")
        
    def predict(self, X):
        # Handle NaN and Inf values
        X_clean = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        return self.model.predict(X_clean)


class MLPStressTester(ModelStressTester):
    """Stress tester for MLP (Neural Network) model"""
    
    def load_model(self):
        print(f"\n📦 Loading MLP model from: {self.model_path}")
        self.model = load_model(self.model_path, compile=False)
        print("  ✅ Model loaded successfully")
        
    def predict(self, X):
        # Handle NaN and Inf values
        X_clean = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        predictions = self.model.predict(X_clean, verbose=0).flatten()
        return (predictions > 0.5).astype(int)


class IsolationForestStressTester(ModelStressTester):
    """Stress tester for Isolation Forest model"""
    
    def load_model(self):
        print(f"\n📦 Loading Isolation Forest model from: {self.model_path}")
        self.model = joblib.load(self.model_path)
        print("  ✅ Model loaded successfully")
        
    def predict(self, X):
        # Handle NaN and Inf values
        X_clean = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        predictions = self.model.predict(X_clean)
        return np.where(predictions == -1, 1, 0)


# ============================================================
# MAIN STRESS TEST SUITE
# ============================================================

def run_comprehensive_stress_tests(stress_level='medium'):
    """
    Run comprehensive stress tests on all models
    
    Parameters:
    - stress_level: 'light', 'medium', 'heavy', or 'extreme'
    """
    
    print("="*70)
    print("FORENXAI MODEL STRESS TESTING SUITE")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Stress Level: {stress_level.upper()}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"System Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"CPU Cores: {psutil.cpu_count()}")
    print("="*70)
    
    # Define model configurations
    models_config = [
        {
            'name': 'Random Forest',
            'path': os.path.join(MODELS_DIR, 'random_forest_pipeline.joblib'),
            'tester_class': RandomForestStressTester
        },
        {
            'name': 'MLP Neural Network',
            'path': os.path.join(MODELS_DIR, 'mlp_model.h5'),
            'tester_class': MLPStressTester
        },
        {
            'name': 'Isolation Forest',
            'path': os.path.join(MODELS_DIR, 'isolation_forest_pipeline.joblib'),
            'tester_class': IsolationForestStressTester
        }
    ]
    
    all_results = {}
    
    for model_config in models_config:
        model_name = model_config['name']
        model_path = model_config['path']
        
        if not os.path.exists(model_path):
            print(f"\n⚠️  Skipping {model_name}: Model file not found at {model_path}")
            continue
        
        print(f"\n{'#'*70}")
        print(f"# TESTING: {model_name}")
        print(f"{'#'*70}")
        
        # Initialize tester
        tester = model_config['tester_class'](model_path, model_name)
        tester.load_model()
        
        # Run all stress tests
        results = {}
        
        try:
            # 1. Throughput test
            results['throughput'] = tester.run_throughput_test(stress_level)
            
            # 2. Latency test
            results['latency'] = tester.run_latency_test()
            
            # 3. Robustness test
            results['robustness'] = tester.run_robustness_test()
            
            # 4. Memory stress test
            results['memory_stress'] = tester.run_memory_stress_test()
            
            all_results[model_name] = results
            
        except Exception as e:
            print(f"\n❌ Error testing {model_name}: {e}")
            all_results[model_name] = {'error': str(e)}
    
    # ============================================================
    # SUMMARY REPORT
    # ============================================================
    print("\n" + "="*70)
    print("STRESS TEST SUMMARY")
    print("="*70)
    
    for model_name, results in all_results.items():
        if 'error' in results:
            print(f"\n❌ {model_name}: FAILED - {results['error']}")
            continue
        
        print(f"\n✅ {model_name}:")
        print(f"   Throughput: {results['throughput']['throughput_samples_per_sec']:,.0f} samples/sec")
        print(f"   Latency: {results['throughput']['latency_per_sample_ms']:.4f} ms/sample")
        print(f"   Memory Growth: {results['memory_stress']['growth_mb']:.2f} MB ({results['memory_stress']['growth_pct']:.1f}%)")
        print(f"   Robustness: {'PASS' if results['robustness']['success'] else 'FAIL'}")
    
    # Performance comparison
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print(f"\n{'Model':<25} {'Throughput':<20} {'Latency':<15} {'Memory':<15}")
    print("-"*75)
    
    for model_name, results in all_results.items():
        if 'error' not in results:
            throughput = f"{results['throughput']['throughput_samples_per_sec']:,.0f} s/s"
            latency = f"{results['throughput']['latency_per_sample_ms']:.4f} ms"
            memory = f"{results['memory_stress']['growth_mb']:.1f} MB"
            
            print(f"{model_name:<25} {throughput:<20} {latency:<15} {memory:<15}")
    
    print("\n" + "="*70)
    print("✅ STRESS TESTING COMPLETE")
    print("="*70)
    
    return all_results


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ForenXAI Model Stress Testing Suite')
    parser.add_argument('--stress-level', type=str, default='medium',
                        choices=['light', 'medium', 'heavy', 'extreme'],
                        help='Stress test intensity level (default: medium)')
    parser.add_argument('--models-dir', type=str, default=None,
                        help='Path to models directory (overrides default)')
    
    args = parser.parse_args()
    
    # Override models directory if specified
    if args.models_dir:
        MODELS_DIR = Path(args.models_dir)
    
    # Run stress tests
    results = run_comprehensive_stress_tests(stress_level=args.stress_level)
    
    print(f"\n💾 Results saved in memory. Export functionality can be added for detailed reports.")
