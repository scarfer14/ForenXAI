"""
Compare Stress Test Results
============================
Runs both synthetic data and random data stress tests,
then compares the results side-by-side.

Useful for:
- Validating that models work with both generated and real data
- Identifying performance differences
- Ensuring production readiness

Usage: python compare_stress_tests.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from stress_test_synthetic_data import run_synthetic_data_stress_tests
from stress_test_models import run_comprehensive_stress_tests

def compare_results(synthetic_results, random_results):
    """Compare and display results from both test types"""
    
    print("\n" + "="*80)
    print("STRESS TEST COMPARISON: SYNTHETIC vs RANDOM DATA")
    print("="*80)
    
    print(f"\n{'Model':<20} {'Data Type':<15} {'Throughput':<18} {'Latency':<15} {'Accuracy/F1':<15}")
    print("-"*83)
    
    # Get common models
    all_models = set(synthetic_results.keys()) | set(random_results.keys())
    
    for model_name in sorted(all_models):
        # Synthetic results
        if model_name in synthetic_results and 'error' not in synthetic_results[model_name]:
            s_res = synthetic_results[model_name]
            print(f"{model_name:<20} {'Synthetic':<15} "
                  f"{s_res['throughput_samples_per_sec']:>8,.0f} s/s     "
                  f"{s_res['latency_per_sample_ms']:>6.4f} ms    "
                  f"F1={s_res['f1_score']:.4f}")
        
        # Random results
        if model_name in random_results and 'error' not in random_results[model_name]:
            r_res = random_results[model_name]['throughput']
            print(f"{'':<20} {'Random Gen':<15} "
                  f"{r_res['throughput_samples_per_sec']:>8,.0f} s/s     "
                  f"{r_res['latency_per_sample_ms']:>6.4f} ms    "
                  f"N/A")
        
        print()
    
    print("="*80)
    print("INSIGHTS")
    print("="*80)
    print("""
✓ Synthetic Data Tests:
  - Uses real network traffic patterns from data/synthetic/
  - Provides accuracy metrics (ground truth available)
  - Tests realistic attack distributions
  - Validates production readiness

✓ Random Data Tests:
  - Uses generated random data
  - Faster execution (no disk I/O)
  - Tests general performance characteristics
  - Good for quick validation

💡 Recommendation:
  - Use Random tests for development and quick checks
  - Use Synthetic tests before deployment and for validation
  - Both should show similar throughput/latency characteristics
    """)
    
    print("="*80)

if __name__ == "__main__":
    print("="*80)
    print("RUNNING COMPARATIVE STRESS TESTS")
    print("="*80)
    print("\nThis will run both test types and compare results...")
    print("Estimated time: 3-5 minutes\n")
    
    # Run synthetic data tests (limited to 10K samples for speed)
    print("\n[1/2] Running Synthetic Data Tests...")
    print("-"*80)
    synthetic_results = run_synthetic_data_stress_tests(
        use_validation=False,
        sample_size=10000
    )
    
    # Run random data tests (medium stress level)
    print("\n[2/2] Running Random Data Tests...")
    print("-"*80)
    random_results = run_comprehensive_stress_tests(stress_level='medium')
    
    # Compare results
    if synthetic_results and random_results:
        compare_results(synthetic_results, random_results)
    else:
        print("\n❌ One or both tests failed. Cannot compare results.")
    
    print("\n✅ Comparison complete!")
