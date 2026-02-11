"""
Quick Stress Test Example
=========================
Simplified script to quickly validate model performance after training.
Perfect for running in Google Colab or local environments.

Usage:
    python quick_stress_test.py
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from stress_test_models import (
    RandomForestStressTester,
    MLPStressTester,
    IsolationForestStressTester,
    generate_synthetic_data
)

def quick_test(model_path, model_name, tester_class):
    """Run a quick performance check on a model"""
    print(f"\n{'='*60}")
    print(f"Quick Test: {model_name}")
    print(f"{'='*60}")
    
    # Initialize tester
    tester = tester_class(model_path, model_name)
    
    try:
        # Load model
        tester.load_model()
        
        # Generate small test dataset
        print("\nGenerating 1,000 test samples...")
        X, y = generate_synthetic_data(1000, n_features=34)
        
        # Test prediction speed
        print("Testing prediction speed...")
        start = time.time()
        predictions = tester.predict(X)
        elapsed = time.time() - start
        
        # Calculate metrics
        throughput = 1000 / elapsed
        latency = (elapsed / 1000) * 1000  # ms per sample
        
        # Results
        print(f"\n✅ Results:")
        print(f"   Predictions: {len(predictions):,}")
        print(f"   Time: {elapsed:.3f} seconds")
        print(f"   Throughput: {throughput:,.0f} samples/sec")
        print(f"   Latency: {latency:.4f} ms/sample")
        print(f"   Status: {'PASS ✓' if throughput > 100 else 'SLOW ⚠️'}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("QUICK STRESS TEST - ForenXAI Models")
    print("="*60)
    
    # Determine models directory
    IN_COLAB = False
    try:
        from google.colab import drive
        IN_COLAB = True
        MODELS_DIR = '/content/drive/MyDrive/Featured Dataset/trained_models'
    except:
        MODELS_DIR = Path(__file__).parent.parent / 'models'
    
    print(f"Models Directory: {MODELS_DIR}\n")
    
    # Test configurations
    tests = [
        {
            'path': f"{MODELS_DIR}/random_forest_pipeline.joblib",
            'name': 'Random Forest',
            'tester': RandomForestStressTester
        },
        {
            'path': f"{MODELS_DIR}/mlp_model.h5",
            'name': 'MLP Neural Network',
            'tester': MLPStressTester
        },
        {
            'path': f"{MODELS_DIR}/isolation_forest_pipeline.joblib",
            'name': 'Isolation Forest',
            'tester': IsolationForestStressTester
        }
    ]
    
    # Run quick tests
    results = []
    for test_config in tests:
        import os
        if os.path.exists(test_config['path']):
            success = quick_test(
                test_config['path'],
                test_config['name'],
                test_config['tester']
            )
            results.append((test_config['name'], success))
        else:
            print(f"\n⚠️  Skipping {test_config['name']}: Model not found")
            results.append((test_config['name'], False))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:<25} {status}")
    
    print(f"\n{'='*60}")
    print("Quick test complete!")
    print("For comprehensive testing, run: python stress_test_models.py")
    print(f"{'='*60}")
