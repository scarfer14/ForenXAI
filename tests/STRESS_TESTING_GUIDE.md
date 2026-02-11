# Stress Testing Guide for ForenXAI Models

## Overview
Comprehensive stress testing suite to evaluate model performance under various load conditions using synthetic datasets.

## Quick Start

### 1. Basic Usage
```bash
# Navigate to tests directory
cd ForenXAI/tests

# Run with default settings (medium stress)
python stress_test_models.py

# Run with different stress levels
python stress_test_models.py --stress-level light    # 1K samples
python stress_test_models.py --stress-level medium   # 10K samples
python stress_test_models.py --stress-level heavy    # 100K samples
python stress_test_models.py --stress-level extreme  # 500K samples
```

### 2. Custom Models Directory
```bash
python stress_test_models.py --models-dir "/path/to/your/models"
```

### 3. Google Colab
```python
# In Colab notebook
!cd /content/drive/MyDrive/ForenXAI/tests
!python stress_test_models.py --stress-level medium
```

## Test Suite Components

### 1. **Throughput Test**
- Measures predictions per second
- Tests with 1K to 500K samples
- Reports: samples/sec, latency per sample, memory usage

### 2. **Latency Test**
- Measures response time across batch sizes (1, 10, 100, 1000)
- 10 iterations per batch size for statistical accuracy
- Reports: average latency, std deviation, latency per sample

### 3. **Robustness Test**
- Tests edge cases:
  - 10% NaN values
  - 10% Infinite values
  - 10% Extreme values (1e10)
  - 10% All zeros
- Validates error handling and prediction stability

### 4. **Memory Stress Test**
- 10 iterations of 50K predictions
- Monitors memory growth and leaks
- Reports: initial, peak, final memory, growth percentage

## Expected Output

```
======================================================================
FORENXAI MODEL STRESS TESTING SUITE
======================================================================
Date: 2026-02-11 15:30:45
Stress Level: MEDIUM
Models Directory: /path/to/models
System Memory: 16.00 GB
CPU Cores: 8
======================================================================

######################################################################
# TESTING: Random Forest
######################################################################

📦 Loading Random Forest model from: /path/to/models/random_forest_pipeline.joblib
  ✅ Model loaded successfully

======================================================================
THROUGHPUT TEST - Random Forest (MEDIUM load)
======================================================================
  Generating 10,000 synthetic samples...
  Processing 10,000 predictions...

  ✅ Results:
     Total Time: 2.45s
     Throughput: 4,082 samples/sec
     Latency: 0.2449 ms/sample
     Memory Delta: 15.23 MB

======================================================================
LATENCY TEST - Random Forest
======================================================================
  Batch Size    1:    1.23 ms ±   0.05 ms (1.2300 ms/sample)
  Batch Size   10:    5.67 ms ±   0.12 ms (0.5670 ms/sample)
  Batch Size  100:   45.89 ms ±   1.23 ms (0.4589 ms/sample)
  Batch Size 1000:  412.34 ms ±   8.91 ms (0.4123 ms/sample)

======================================================================
STRESS TEST SUMMARY
======================================================================

✅ Random Forest:
   Throughput: 4,082 samples/sec
   Latency: 0.2449 ms/sample
   Memory Growth: 12.45 MB (5.2%)
   Robustness: PASS

✅ MLP Neural Network:
   Throughput: 8,521 samples/sec
   Latency: 0.1173 ms/sample
   Memory Growth: 25.67 MB (8.1%)
   Robustness: PASS
```

## Performance Benchmarks

### Recommended Benchmarks (Medium Load - 10K samples)

| Model | Throughput | Latency | Memory Growth |
|-------|------------|---------|---------------|
| Random Forest | 3,000-5,000 s/s | < 0.3 ms | < 20 MB |
| MLP Neural Net | 7,000-10,000 s/s | < 0.15 ms | < 30 MB |
| Isolation Forest | 2,000-4,000 s/s | < 0.5 ms | < 15 MB |

### ⚠️ Warning Thresholds
- **Throughput**: < 500 samples/sec (too slow for production)
- **Latency**: > 10 ms/sample (unacceptable for real-time)
- **Memory Growth**: > 100 MB (potential memory leak)

## Stress Level Guide

### Light (1,000 samples)
- **Purpose**: Quick smoke test, development verification
- **Time**: ~10-20 seconds
- **Use When**: Testing new features, debugging

### Medium (10,000 samples)
- **Purpose**: Standard CI/CD testing, performance baseline
- **Time**: ~30-60 seconds
- **Use When**: Pre-deployment checks, regression testing

### Heavy (100,000 samples)
- **Purpose**: Production readiness, load testing
- **Time**: ~2-5 minutes
- **Use When**: Before major releases, capacity planning

### Extreme (500,000 samples)
- **Purpose**: Stress testing, breaking point analysis
- **Time**: ~10-20 minutes
- **Use When**: Infrastructure validation, worst-case scenarios

## Troubleshooting

### Issue: "Model file not found"
```bash
# Verify models directory
ls -la ../models/

# Or specify custom path
python stress_test_models.py --models-dir "/full/path/to/models"
```

### Issue: "Out of memory"
- Reduce stress level: `--stress-level light`
- Close other applications
- Upgrade system RAM (minimum 8GB recommended)

### Issue: Tests failing on edge cases
- Expected behavior: Models handle NaN/Inf automatically
- Check model training included robust preprocessing
- Review data cleaning in train_ml_models.py

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: Model Stress Testing
on: [push, pull_request]

jobs:
  stress-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run stress tests
        run: python tests/stress_test_models.py --stress-level medium
```

## Advanced Usage

### Programmatic Usage
```python
from stress_test_models import run_comprehensive_stress_tests

# Run tests programmatically
results = run_comprehensive_stress_tests(stress_level='heavy')

# Access results
rf_throughput = results['Random Forest']['throughput']['throughput_samples_per_sec']
print(f"Random Forest: {rf_throughput:,.0f} samples/sec")
```

### Custom Synthetic Data
```python
from stress_test_models import generate_synthetic_data

# Generate custom test data
X, y = generate_synthetic_data(n_samples=50000, n_features=34, anomaly_ratio=0.2)

# Use with your models
predictions = model.predict(X)
```

## Dependencies

Ensure these packages are installed:
```bash
pip install numpy pandas scikit-learn tensorflow joblib psutil
```

## Output Files

Currently, results are displayed in console only. Future versions will support:
- JSON export for automated analysis
- CSV reports for Excel import
- HTML dashboards for visualization
- Performance trend tracking over time

## Best Practices

1. **Run before deployment**: Always stress test before production
2. **Automate**: Include in CI/CD pipeline
3. **Track trends**: Monitor performance over time
4. **Test realistic loads**: Use stress levels matching expected traffic
5. **Document baselines**: Record acceptable performance thresholds

## Support

For issues or questions:
- Check the main README.md
- Review model training documentation in docs/
- Verify models are properly trained and saved

---

**Created by**: ForenXAI Team  
**Last Updated**: February 2026  
**Version**: 1.0.0
