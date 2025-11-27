# CSV Data Files Guide for Cross-Validation

## ✅ YES - CSV Files Are VERY Useful for Cross-Validation!

Cross-validation requires your actual dataset, so CSV files are **essential**. Here's why:

### Why CSV Files Matter for Cross-Validation

1. **K-Fold Splitting**: Cross-validation splits your CSV data into k parts (e.g., 10 folds)
2. **Training & Testing**: Each fold becomes a test set while others train the model
3. **Robust Evaluation**: Tests model performance on different data subsets
4. **Confidence Intervals**: Provides statistical confidence in your results

## 📁 Your Current CSV Files

Located in: `Multiple-Disease-Prediction-Webapp/Frontend/data/`

Available:
- ✅ `lung_cancer.csv` - Can be added to analysis
- ✅ `dataset.csv` - General symptom dataset
- ⚠️ Missing: diabetes.csv, heart.csv, parkinsons.csv, etc.

## 🔧 How to Add New CSV Files

### Method 1: Add to Existing Scripts (Already Done!)

I've updated both scripts to include `lung_cancer.csv`:

**cross_validation_analysis.py** - Now includes lung cancer
**hyperparameter_tuning_analysis.py** - Now includes lung cancer

### Method 2: Add Your Own CSV File

```python
# In the load_disease_data() method, add:
data_paths = {
    'your_disease': 'path/to/your_file.csv',
    # ... other diseases
}
```

**Requirements for CSV format:**
- Last column = target/label (0 or 1 for disease prediction)
- All other columns = features
- No missing values (or handle them first)
- Numeric data (categorical should be encoded)

### Method 3: Use dataset.csv

If `dataset.csv` contains symptom-based predictions:

```python
# Add to data_paths:
'symptoms': 'Multiple-Disease-Prediction-Webapp/Frontend/data/dataset.csv'
```

## 🚀 Running Cross-Validation

```bash
# Run cross-validation on all diseases (including lung cancer now)
python cross_validation_analysis.py

# Run hyperparameter tuning
python hyperparameter_tuning_analysis.py
```

## 📊 What You'll Get

After running with your CSV files:

1. **cv_results_[disease].json** - Detailed cross-validation metrics
2. **Confidence intervals** - Statistical reliability (95% CI)
3. **Fold-by-fold scores** - Performance across all splits
4. **Model comparison** - RF vs XGBoost vs SVM

## ⚠️ Important Notes

### CSV File Structure Expected:
```
feature1, feature2, feature3, ..., target
1.2,      3.4,      5.6,      ..., 0
2.3,      4.5,      6.7,      ..., 1
```

### If Your CSV Has Different Structure:

Modify the `load_disease_data()` method:

```python
def load_disease_data(self, disease_name):
    df = pd.read_csv(data_paths[disease_name])
    
    # Option 1: Target is named column
    X = df.drop('disease_column_name', axis=1).values
    y = df['disease_column_name'].values
    
    # Option 2: Target is first column
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    
    # Option 3: Target is last column (current default)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    return X, y, df.columns[:-1].tolist()
```

## 🎯 Quick Test

To test if your CSV works:

```python
import pandas as pd

# Load your CSV
df = pd.read_csv('Multiple-Disease-Prediction-Webapp/Frontend/data/lung_cancer.csv')

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("First few rows:")
print(df.head())
print("\nTarget distribution:")
print(df.iloc[:, -1].value_counts())
```

## 💡 Pro Tips

1. **More data = better cross-validation**: Larger CSV files give more reliable results
2. **Balanced classes**: Check if your target has similar counts of 0s and 1s
3. **Feature scaling**: Some models (SVM) benefit from normalized features
4. **Missing values**: Handle them before cross-validation

## 🔍 Troubleshooting

**Error: "File not found"**
- Check the file path is correct
- Ensure CSV is in the data folder

**Error: "could not convert string to float"**
- Your CSV has non-numeric data
- Encode categorical variables first

**Low accuracy scores**
- Check data quality
- Try different preprocessing
- Verify target column is correct

## Next Steps

1. ✅ I've added lung_cancer.csv to your scripts
2. Run the scripts to see results
3. Check if you need the other missing CSV files (diabetes, heart, etc.)
4. If you have them elsewhere, copy them to the data folder
