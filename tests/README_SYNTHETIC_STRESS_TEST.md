# Synthetic Data Stress Testing

## Overview
Professional stress testing using **real synthetic network traffic data** from the `data/synthetic/` folder. Tests trained models with realistic attack patterns and ground truth labels for comprehensive evaluation.

## Quick Start

### Basic Usage
```bash
cd ForenXAI/tests

# Test with training data (44K samples)
python stress_test_synthetic_data.py

# Test with validation data
python stress_test_synthetic_data.py --validation

# Test with limited samples (faster)
python stress_test_synthetic_data.py --sample-size 5000
```

### Advanced Options
```bash
# Custom data directory
python stress_test_synthetic_data.py --data-dir "/path/to/data/synthetic"

# Custom models directory
python stress_test_synthetic_data.py --models-dir "/path/to/models"

# Combine options
python stress_test_synthetic_data.py --validation --sample-size 10000
```

## What It Tests

### 1. **Performance Metrics**
- ⚡ **Throughput**: Predictions per second
- ⏱️ **Latency**: Milliseconds per prediction
- 💾 **Memory**: Memory consumption during prediction
- 🖥️ **CPU**: Average CPU usage

### 2. **Accuracy Metrics** (with ground truth)
- 🎯 **Accuracy**: Overall correctness
- 📊 **Precision**: Attack detection precision
- 🔍 **Recall**: Attack detection coverage
- ⚖️ **F1-Score**: Balanced performance measure
- 📈 **Confusion Matrix**: TN, FP, FN, TP breakdown

### 3. **Attack Type Analysis**
- Performance breakdown by attack category:
  - Benign (normal traffic)
  - DDoS attacks
  - Exploits
  - Fuzzers
  - Generic attacks
  - Reconnaissance
  - Shellcode

### 4. **Stress Levels**
- Tests scalability with varying sample sizes:
  - **Mini**: 1,000 samples
  - **Small**: 5,000 samples
  - **Medium**: 10,000 samples
  - **Large**: 20,000 samples

## Expected Output

```
======================================================================
SYNTHETIC DATA STRESS TESTING SUITE
======================================================================
Date: 2026-02-11 16:45:30
Data Directory: /path/to/data/synthetic
Models Directory: /path/to/models
System Memory: 16.00 GB
CPU Cores: 8
======================================================================

📂 Loading synthetic data from: synthetic_train_split.csv
  Loading data from: data/synthetic/synthetic_train_split.csv
  Loaded 44,447 samples with 34 features
  Label distribution: Normal=42,103 (94.7%), Attack=2,344 (5.3%)

######################################################################
# TESTING: Random Forest
######################################################################
  📦 Loading Random Forest model...
  ✅ Model loaded successfully

======================================================================
PERFORMANCE TEST - Random Forest
======================================================================

  Processing 44,447 predictions...

  ⚡ PERFORMANCE METRICS:
     Total Time: 8.234 seconds
     Throughput: 5,398 samples/sec
     Latency: 0.1853 ms/sample
     Memory Delta: 18.45 MB
     CPU Usage: 42.3%

  📊 ACCURACY METRICS:
     Accuracy:  0.9842 (98.42%)
     Precision: 0.9523
     Recall:    0.8976
     F1-Score:  0.9241

  🎯 CONFUSION MATRIX:
     True Negatives:  41,521 (correctly identified normal)
     False Positives: 582 (normal flagged as attack)
     False Negatives: 240 (missed attacks)
     True Positives:  2,104 (correctly detected attacks)

  🔍 ATTACK TYPE BREAKDOWN:
     Benign              : 0.9862 (42,103 samples)
     DDoS                : 0.9345 (856 samples)
     Exploits            : 0.9123 (421 samples)
     Fuzzers             : 0.8876 (312 samples)
     Generic             : 0.9201 (389 samples)
     Reconnaissance      : 0.8734 (201 samples)
     Shellcode           : 0.9012 (165 samples)

======================================================================
STRESS LEVELS TEST - Random Forest
======================================================================

  [MINI] Testing with 1,000 samples...
     Time: 0.187s | Throughput: 5,348 s/s | Accuracy: 0.9840

  [SMALL] Testing with 5,000 samples...
     Time: 0.921s | Throughput: 5,429 s/s | Accuracy: 0.9836

  [MEDIUM] Testing with 10,000 samples...
     Time: 1.842s | Throughput: 5,429 s/s | Accuracy: 0.9844

  [LARGE] Testing with 20,000 samples...
     Time: 3.687s | Throughput: 5,424 s/s | Accuracy: 0.9841

======================================================================
STRESS TEST SUMMARY - SYNTHETIC DATA
======================================================================
Dataset: synthetic_train_split.csv (44,447 samples)
Attack Types: 7 unique types
======================================================================

Model                     Throughput           Latency         Accuracy     F1-Score    
------------------------------------------------------------------------------------
Random Forest             5,398 s/s            0.1853 ms       0.9842       0.9241      
MLP Neural Network        12,456 s/s           0.0803 ms       0.9867       0.9312      
Isolation Forest          3,234 s/s            0.3092 ms       0.9523       0.8876      

======================================================================
BEST PERFORMERS
======================================================================
🎯 Best Accuracy:  MLP Neural Network (0.9867)
⚡ Best Throughput: MLP Neural Network (12,456 s/s)
📊 Best F1-Score:  MLP Neural Network (0.9312)

======================================================================
✅ SYNTHETIC DATA STRESS TESTING COMPLETE
======================================================================
```

## Data Files

### Training Split
- **File**: `synthetic_train_split.csv`
- **Size**: ~44,447 samples
- **Use**: Primary stress testing and performance validation

### Validation Split
- **File**: `synthetic_val_split.csv`
- **Size**: Validation subset
- **Use**: Independent performance verification

## Features Tested

The synthetic data includes 34 network traffic features:
- Protocol information (PROTOCOL, L7_PROTO)
- Packet statistics (IN_BYTES, OUT_BYTES, IN_PKTS)
- Flow characteristics (FLOW_DURATION_MILLISECONDS, TCP_FLAGS)
- Timing features (IAT min/max/avg/stddev)
- Packet size distributions (NUM_PKTS_UP_TO_128_BYTES, etc.)
- TCP window sizes, DNS queries, ICMP types
- And more...

## Performance Benchmarks

### Expected Performance (44K samples)

| Model | Throughput | Latency | Accuracy | F1-Score |
|-------|------------|---------|----------|----------|
| **Random Forest** | 4,000-6,000 s/s | 0.15-0.25 ms | > 0.98 | > 0.92 |
| **MLP Neural Net** | 10,000-15,000 s/s | 0.06-0.10 ms | > 0.98 | > 0.93 |
| **Isolation Forest** | 2,000-4,000 s/s | 0.25-0.50 ms | > 0.94 | > 0.88 |

### Quality Thresholds

✅ **PASS**: Accuracy > 95%, F1-Score > 0.85  
⚠️ **WARNING**: Accuracy 90-95%, F1-Score 0.75-0.85  
❌ **FAIL**: Accuracy < 90%, F1-Score < 0.75

## Advantages Over Random Testing

### ✅ Real Attack Patterns
- Uses actual synthetic network traffic patterns
- Realistic attack distributions (5-10% attack rate)
- Multiple attack types tested simultaneously

### ✅ Ground Truth Validation
- Compares predictions against known labels
- Provides accuracy, precision, recall, F1-score
- Confusion matrix for error analysis

### ✅ Attack-Specific Insights
- Performance breakdown by attack category
- Identifies which attacks are harder to detect
- Helps tune models for specific threats

### ✅ Production Readiness
- Tests on real-world data distributions
- Validates model generalization
- Ensures models work with actual dataset characteristics

## Troubleshooting

### Issue: "Data file not found"
```bash
# Verify data directory
ls -la data/synthetic/

# Or specify custom path
python stress_test_synthetic_data.py --data-dir "/path/to/data/synthetic"
```

### Issue: "Model file not found"
```bash
# Check models directory
ls -la models/

# Train models first if missing
python src/backend/train_ml_models.py
```

### Issue: Memory errors with large datasets
```bash
# Use smaller sample size
python stress_test_synthetic_data.py --sample-size 10000
```

### Issue: Slow performance
- Close other applications
- Use `--sample-size` for faster testing
- Ensure models are properly trained and saved

## Integration with CI/CD

```yaml
# GitHub Actions example
- name: Synthetic Data Stress Test
  run: |
    cd ForenXAI/tests
    python stress_test_synthetic_data.py --sample-size 5000
```

## Comparison with Other Tests

| Test Type | Data Source | Purpose | Speed |
|-----------|-------------|---------|-------|
| **stress_test_models.py** | Generated random | General performance | Fast |
| **stress_test_synthetic_data.py** | Real synthetic files | Production validation | Medium |
| **quick_stress_test.py** | Generated random | Quick smoke test | Very fast |

## Best Practices

1. **Run before deployment**: Validate with real data patterns
2. **Test both splits**: Training for performance, validation for generalization
3. **Monitor attack types**: Ensure all attack categories are detected
4. **Track trends**: Compare results across model versions
5. **Document baselines**: Record acceptable performance thresholds

## Output Files

Currently displays results in console. Future enhancements:
- JSON export for automated analysis
- CSV reports for performance tracking
- HTML dashboard with visualizations
- Attack-type specific reports

---

**Created by**: ForenXAI Team  
**Last Updated**: February 2026  
**Version**: 1.0.0  
**Data Source**: `data/synthetic/` (real network traffic patterns)
