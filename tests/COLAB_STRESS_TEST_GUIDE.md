# Google Colab Stress Testing Guide

## 🚀 Quick Start

### Step 1: Upload Files to Google Drive

Upload these files to your Google Drive:

```
My Drive/
└── Featured Dataset/
    ├── trained_models/
    │   ├── random_forest_pipeline.joblib
    │   ├── mlp_model.h5
    │   └── isolation_forest_pipeline.joblib
    └── processed/
        └── synthetic_train_split.csv
```

### Step 2: Upload CSV

**You already have the CSV!** Upload `data/synthetic/synthetic_train_split.csv` to:
```
My Drive/Featured Dataset/processed/
```

### Step 3: Open Notebook in Colab

1. Upload `Colab_Stress_Test.ipynb` to Google Colab
2. Or open directly: File → Upload notebook → Select the .ipynb file

### Step 4: Run All Cells

Click: **Runtime → Run all**

## 📊 What You'll Get

### Performance Metrics:
- ⚡ **Throughput**: How many predictions per second (target: >5,000)
- ⏱️ **Latency**: Time per prediction in milliseconds (target: <0.5ms)
- 💾 **Memory**: RAM usage during testing

### Accuracy Metrics:
- 🎯 **Accuracy**: Overall correctness (target: >95%)
- 📊 **Precision**: How many predicted attacks were real attacks
- 🔍 **Recall**: How many real attacks were detected
- ⚖️ **F1-Score**: Balanced metric (target: >0.90)

### Detailed Analysis:
- Confusion matrix (TP, TN, FP, FN)
- Attack type breakdown (DDoS, Exploits, etc.)
- Model comparison table
- Best performer identification

## 🎯 Sample Output

```
================================================================================
TESTING: Random Forest
================================================================================

⚡ PERFORMANCE:
   Time: 1.234 seconds
   Throughput: 8,104 samples/sec
   Latency: 0.1234 ms/sample
   Memory: 45.23 MB

📊 ACCURACY:
   Accuracy:  0.9845 (98.45%)
   Precision: 0.9423
   Recall:    0.8912
   F1-Score:  0.9160

🎯 CONFUSION MATRIX:
   True Negatives:  9,412
   False Positives: 58
   False Negatives: 64
   True Positives:  466

🔍 ATTACK TYPE BREAKDOWN:
   Benign              : 0.9940 (9,470 samples)
   DDoS                : 0.8750 (120 samples)
   Exploits            : 0.9200 (150 samples)
   ...
```

## ⚙️ Configuration Options

### Test with Full Dataset (44K samples):
```python
SAMPLE_SIZE = None  # Cell 4
```

### Test with Smaller Sample (faster):
```python
SAMPLE_SIZE = 10000  # Default - takes ~2 minutes
```

### Use Validation Split Instead:
```python
TEST_CSV = 'synthetic_val_split.csv'  # Cell 2
```

## ✅ Success Criteria

Your models are **production-ready** if:
- ✅ Accuracy > 95%
- ✅ F1-Score > 0.85
- ✅ Throughput > 1,000 samples/sec
- ✅ False Negatives < 10% (high recall)

## 🔧 Troubleshooting

### Error: "CSV file not found"
**Solution**: Upload `synthetic_train_split.csv` to `My Drive/Featured Dataset/processed/`

### Error: "Model not found"
**Solution**: Wait for training to complete, then upload models to `My Drive/Featured Dataset/trained_models/`

### Error: "Drive not mounted"
**Solution**: Run Cell 1 again and grant permissions

### Low accuracy (<90%)
**Solution**: Models may need retraining with more data

### Slow performance (<1000 s/s)
**Solution**: 
- Use GPU runtime (Runtime → Change runtime type → GPU)
- Reduce SAMPLE_SIZE for faster testing

## 📈 Expected Runtimes

| Sample Size | Runtime (GPU) | Runtime (CPU) |
|-------------|---------------|---------------|
| 1,000       | ~5 seconds    | ~15 seconds   |
| 10,000      | ~30 seconds   | ~2 minutes    |
| 44,000 (full)| ~2 minutes   | ~8 minutes    |

## 💡 Pro Tips

1. **Use GPU**: Set Runtime → Change runtime type → GPU (T4)
2. **Start Small**: Test with 1,000 samples first
3. **Save Results**: Results auto-save to `stress_test_results.csv`
4. **Rerun Anytime**: Keep the notebook - test future model versions
5. **Compare Models**: Run multiple times with different CSVs

## 📁 File Structure

```
Google Drive Structure:
My Drive/
├── Featured Dataset/
│   ├── trained_models/          # Upload your models here
│   │   ├── random_forest_pipeline.joblib
│   │   ├── mlp_model.h5
│   │   └── isolation_forest_pipeline.joblib
│   └── processed/               # Upload CSV here
│       └── synthetic_train_split.csv
└── stress_test_results.csv      # Auto-created after testing
```

## 🎓 Understanding the Results

### Confusion Matrix:
- **True Positives (TP)**: Correctly detected attacks ✅
- **True Negatives (TN)**: Correctly identified normal traffic ✅
- **False Positives (FP)**: False alarms ⚠️
- **False Negatives (FN)**: Missed attacks ❌ (CRITICAL!)

### What Each Model Tests:
- **Random Forest**: Main classifier (best for accuracy)
- **MLP Neural Network**: Deep learning model (best for complex patterns)
- **Isolation Forest**: Anomaly detection (finds unknown attacks)

### Attack Type Breakdown:
Shows how well each model detects specific attack types:
- **Benign**: Normal traffic
- **DDoS**: Denial of Service attacks
- **Exploits**: Software vulnerability exploits
- **Fuzzers**: Input fuzzing attacks
- **Generic**: Generic malicious traffic
- **Reconnaissance**: Network scanning
- **Shellcode**: Code injection attacks

## 🚀 Next Steps

After stress testing:

1. **If All Pass** (>95% accuracy):
   - ✅ Models are production-ready
   - Deploy to your dashboard
   
2. **If Some Fail** (<90% accuracy):
   - 🔄 Retrain with more epochs
   - 📊 Add more training data
   - ⚙️ Adjust hyperparameters

3. **If Throughput Low** (<1000 s/s):
   - 🎯 Use GPU runtime
   - 🔧 Optimize model architecture
   - 💻 Deploy on better hardware

---

**Need Help?**
- Check cell outputs for specific error messages
- Verify file paths in Cell 2
- Make sure Google Drive is mounted (Cell 1)
- Try with smaller SAMPLE_SIZE first (Cell 4)
