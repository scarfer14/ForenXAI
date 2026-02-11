# ForenXAI Stress Testing Suite - Complete Guide

## 📚 Overview

Professional stress testing suite with **four testing approaches** for comprehensive model validation:

1. **Synthetic Data Testing** (Production validation)
2. **Random Data Testing** (Performance benchmarking)
3. **Quick Testing** (Development smoke tests)
4. **Comparative Analysis** (Side-by-side evaluation)

---

## 🎯 Which Test Should I Use?

### Use **Synthetic Data Test** when:
- ✅ Before deployment to production
- ✅ Validating model accuracy on real patterns
- ✅ Testing with realistic attack distributions
- ✅ Need ground truth metrics (accuracy, precision, recall)
- ✅ Analyzing performance by attack type

### Use **Random Data Test** when:
- ✅ Quick performance checks during development
- ✅ Testing throughput and latency characteristics
- ✅ Validating error handling and robustness
- ✅ No real data available yet
- ✅ CI/CD pipeline automation

### Use **Quick Test** when:
- ✅ After training models (smoke test)
- ✅ Verifying model loads correctly
- ✅ Need results in < 30 seconds
- ✅ Basic "does it work?" validation

### Use **Comparative Test** when:
- ✅ Comparing real vs synthetic data performance
- ✅ Validating consistency across test types
- ✅ Documenting comprehensive results
- ✅ Pre-release quality assurance

---

## 📁 Test Files

| File | Purpose | Speed | Data Source |
|------|---------|-------|-------------|
| [stress_test_synthetic_data.py](stress_test_synthetic_data.py) | Production validation | Medium | Real synthetic files |
| [stress_test_models.py](stress_test_models.py) | Performance benchmark | Fast | Generated random |
| [quick_stress_test.py](quick_stress_test.py) | Smoke test | Very Fast | Generated random |
| [compare_stress_tests.py](compare_stress_tests.py) | Side-by-side analysis | Medium | Both sources |

---

## 🚀 Quick Start Commands

### 1. Synthetic Data Test (Recommended for Validation)
```bash
cd ForenXAI/tests

# Full test with all training data (44K samples)
python stress_test_synthetic_data.py

# Quick test (5K samples) - faster
python stress_test_synthetic_data.py --sample-size 5000

# Test validation split
python stress_test_synthetic_data.py --validation
```

**Output**: Accuracy, F1-score, confusion matrix, attack-type breakdown

---

### 2. Random Data Test (Good for Development)
```bash
# Medium stress (10K samples)
python stress_test_models.py --stress-level medium

# Light stress (1K samples) - fastest
python stress_test_models.py --stress-level light

# Heavy stress (100K samples) - thorough
python stress_test_models.py --stress-level heavy
```

**Output**: Throughput, latency, memory usage, robustness

---

### 3. Quick Test (Smoke Test)
```bash
# 30-second validation
python quick_stress_test.py
```

**Output**: Basic pass/fail for each model

---

### 4. Comparative Test (Full Analysis)
```bash
# Compare both test types
python compare_stress_tests.py
```

**Output**: Side-by-side comparison table

---

## 📊 What Each Test Provides

### Synthetic Data Test
```
✓ Accuracy: 0.9842 (98.42%)
✓ Precision: 0.9523
✓ Recall: 0.8976
✓ F1-Score: 0.9241
✓ Throughput: 5,398 samples/sec
✓ Latency: 0.1853 ms/sample
✓ Attack breakdown: Performance by attack type
✓ Confusion matrix: TN, FP, FN, TP
```

### Random Data Test
```
✓ Throughput: 4,082 samples/sec
✓ Latency: 0.2449 ms/sample
✓ Memory usage: 15.23 MB delta
✓ Robustness: PASS (handles NaN, Inf, extreme values)
✓ Memory stability: No leaks detected
✓ Stress levels: Performance across sample sizes
```

---

## 🎓 Understanding the Results

### Performance Metrics

**Throughput** (samples/sec)
- **Excellent**: > 10,000 s/s
- **Good**: 5,000 - 10,000 s/s
- **Acceptable**: 1,000 - 5,000 s/s
- **Poor**: < 1,000 s/s

**Latency** (ms/sample)
- **Excellent**: < 0.1 ms
- **Good**: 0.1 - 0.5 ms
- **Acceptable**: 0.5 - 1.0 ms
- **Poor**: > 1.0 ms

### Accuracy Metrics (Synthetic Data Only)

**F1-Score**
- **Excellent**: > 0.95
- **Good**: 0.90 - 0.95
- **Acceptable**: 0.85 - 0.90
- **Poor**: < 0.85

**Accuracy**
- **Excellent**: > 98%
- **Good**: 95% - 98%
- **Acceptable**: 90% - 95%
- **Poor**: < 90%

---

## 💡 Best Practices

### Before Deployment
```bash
# 1. Quick smoke test
python quick_stress_test.py

# 2. Validate with synthetic data
python stress_test_synthetic_data.py --sample-size 10000

# 3. Run full synthetic test
python stress_test_synthetic_data.py

# 4. Document results
python compare_stress_tests.py > stress_test_report.txt
```

### During Development
```bash
# After model changes
python quick_stress_test.py

# After training
python stress_test_models.py --stress-level light
```

### For CI/CD Pipeline
```bash
# Fast validation (2-3 minutes)
python stress_test_models.py --stress-level medium

# Or with synthetic data
python stress_test_synthetic_data.py --sample-size 5000
```

---

## 🔧 Troubleshooting

### Models Not Found
```bash
# Train models first
cd ForenXAI/src/backend
python train_ml_models.py

# Then run tests
cd ../../tests
python stress_test_synthetic_data.py
```

### Synthetic Data Not Found
```bash
# Check data directory
ls -la ../data/synthetic/

# Should see:
# synthetic_train_split.csv
# synthetic_val_split.csv

# If missing, run data preparation
cd ../src/backend
python feature_prep.py
```

### Out of Memory
```bash
# Reduce sample size
python stress_test_synthetic_data.py --sample-size 5000

# Or use light stress level
python stress_test_models.py --stress-level light
```

---

## 📈 Sample Workflow

### New Model Version
1. Train models: `python train_ml_models.py`
2. Quick test: `python quick_stress_test.py`
3. Synthetic validation: `python stress_test_synthetic_data.py --sample-size 10000`
4. Full test (if looks good): `python stress_test_synthetic_data.py`
5. Document: Save results for comparison

### Before Production Deploy
1. Full synthetic test: `python stress_test_synthetic_data.py`
2. Validation split: `python stress_test_synthetic_data.py --validation`
3. Compare results: `python compare_stress_tests.py`
4. Verify all metrics pass thresholds
5. Deploy with confidence ✅

---

## 📝 Documentation Files

- [README_SYNTHETIC_STRESS_TEST.md](README_SYNTHETIC_STRESS_TEST.md) - Detailed synthetic test guide
- [STRESS_TESTING_GUIDE.md](STRESS_TESTING_GUIDE.md) - Random data test guide
- This file - Complete overview

---

## ✅ Success Criteria

Before deployment, ensure:
- ✓ All models load successfully
- ✓ **Synthetic test** accuracy > 95%
- ✓ **Synthetic test** F1-score > 0.85
- ✓ Throughput > 1,000 samples/sec
- ✓ Latency < 1 ms/sample
- ✓ No memory leaks detected
- ✓ Robustness tests pass
- ✓ Validation split performs similarly to training

---

## 🎯 Recommended Testing Schedule

**Daily** (during development)
- Quick stress test after code changes

**Weekly** (during active development)
- Random data stress test (medium level)
- Synthetic data test (10K samples)

**Before Each Release**
- Full synthetic data test (all data)
- Validation split test
- Comparative analysis
- Document baseline metrics

**After Deployment**
- Monitor production metrics
- Compare with stress test baselines
- Re-run if anomalies detected

---

## 📞 Support

For questions or issues:
1. Check this guide and related documentation
2. Verify models are trained: `ls -la ../models/`
3. Verify data exists: `ls -la ../data/synthetic/`
4. Review error messages carefully
5. Check system resources (RAM, disk space)

---

**Created**: February 2026  
**Team**: ForenXAI  
**Version**: 1.0.0  
**Status**: Production Ready ✅
