# 📍 Where to See All Results

## 🌐 1. IN THE WEB APPLICATION (Recommended)

### Access: http://localhost:8501

### Navigate to These Sections:

#### A. **Model Comparison** (Sidebar Menu)
**What you'll see:**
- Performance comparison table for all 6 diseases
- Accuracy bar charts
- All metrics visualization (Accuracy, Precision, Recall, F1)
- Best performing models

**How to access:**
1. Open http://localhost:8501
2. Click "Model Comparison" in the sidebar
3. View all model performance metrics

---

#### B. **Research Analysis** (Sidebar Menu)
**What you'll see:**
- Cross-Validation Analysis results
- SHAP Explainable AI Analysis
- Hyperparameter Tuning documentation

**How to access:**
1. Click "Research Analysis" in sidebar
2. Select analysis type from dropdown:
   - Cross-Validation Analysis
   - SHAP XAI Analysis
   - Hyperparameter Tuning
   - All Analyses

**Cross-Validation Results:**
- Select diseases to analyze
- Click "Run Cross-Validation Analysis"
- See results with confidence intervals
- Download results as JSON

**SHAP Analysis:**
- Select diseases for SHAP analysis
- Click "Run SHAP Analysis"
- View feature importance plots
- See SHAP summary visualizations

---

#### C. **Individual Disease Prediction Pages**
**What you'll see:**
- Make predictions for specific patients
- View feature importance for that prediction
- See risk factor analysis
- Get personalized recommendations

**How to access:**
1. Click any disease in sidebar:
   - Diabetes Prediction
   - Heart Disease Prediction
   - Parkinson's Prediction
   - Liver Prediction
   - Hepatitis Prediction
   - Chronic Kidney Prediction
2. Enter patient data
3. Click "Predict"
4. View results with explanations

---

## 📁 2. IN FILES (For Research Papers)

### A. Cross-Validation Results

**Location:** `Multiple-Disease-Prediction-Webapp/Frontend/`

**Files:**
```
cv_results_diabetes.json      (6.2 KB)
cv_results_heart.json          (6.2 KB)
cv_results_liver.json          (6.2 KB)
cv_results_hepatitis.json      (4.9 KB)
cv_results_kidney.json         (4.8 KB)
cv_results_lung_cancer.json    (6.2 KB)
```

**What's inside:**
- 10-fold cross-validation results
- Mean accuracy, precision, recall, F1 score
- Standard deviation
- 95% confidence intervals
- Individual fold scores
- ANOVA statistical tests

**How to open:**
1. Navigate to `Multiple-Disease-Prediction-Webapp/Frontend/`
2. Open any `cv_results_*.json` file
3. Use JSON viewer or text editor

**Example content:**
```json
{
  "Random Forest": {
    "accuracy": {
      "mean": 0.7577,
      "std": 0.0431,
      "ci_95_lower": 0.7310,
      "ci_95_upper": 0.7844,
      "fold_scores": [0.792, 0.779, ...]
    }
  }
}
```

---

### B. SHAP Analysis Results

**Location:** `Multiple-Disease-Prediction-Webapp/Frontend/`

**Files (if generated):**
```
shap_summary_diabetes.png
shap_importance_diabetes.png
shap_dependence_diabetes.png
(Similar files for other diseases)
```

**What's inside:**
- SHAP summary plots
- Feature importance visualizations
- Dependence plots

**How to view:**
1. Navigate to `Multiple-Disease-Prediction-Webapp/Frontend/`
2. Look for `shap_*.png` files
3. Open with image viewer
4. Or view in the application's Research Analysis section

---

### C. Model Performance Metrics

**Location:** `Multiple-Disease-Prediction-Webapp/Frontend/models/`

**Files:**
```
all_metrics_summary.json
diabetes_model_metrics.json
heart_disease_model_metrics.json
parkinsons_model_metrics.json
liver_model_metrics.json
hepatitis_model_metrics.json
chronic_model_metrics.json
```

**What's inside:**
```json
{
  "accuracy": 0.758,
  "precision": 0.755,
  "recall": 0.758,
  "f1_score": 0.755
}
```

**How to open:**
1. Navigate to `Multiple-Disease-Prediction-Webapp/Frontend/models/`
2. Open any `*_metrics.json` file
3. Use JSON viewer or text editor

---

## 📊 3. QUICK VIEW COMMANDS

### View Cross-Validation Results:
```powershell
# In PowerShell
Get-Content Multiple-Disease-Prediction-Webapp\Frontend\cv_results_diabetes.json | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

### List All Result Files:
```powershell
dir Multiple-Disease-Prediction-Webapp\Frontend\cv_results_*.json
dir Multiple-Disease-Prediction-Webapp\Frontend\shap_*.png
dir Multiple-Disease-Prediction-Webapp\Frontend\models\*metrics*.json
```

### Open Results Folder:
```powershell
explorer Multiple-Disease-Prediction-Webapp\Frontend
```

---

## 🎯 4. WHAT TO USE FOR WHAT

### For Research Paper:
✅ **Use:** JSON files (`cv_results_*.json`, `*_metrics.json`)
- Copy metrics directly into tables
- Reference confidence intervals
- Cite statistical tests

### For Presentations:
✅ **Use:** Application screenshots
- Model Comparison charts
- SHAP visualizations
- Performance metrics displays

### For Clinical Use:
✅ **Use:** Individual disease prediction pages
- Enter patient data
- Get predictions with explanations
- View risk factors

### For Model Validation:
✅ **Use:** Cross-validation results
- Check confidence intervals
- Verify statistical significance
- Compare model performance

---

## 📝 5. EXAMPLE: HOW TO VIEW DIABETES RESULTS

### In Application:
1. Open http://localhost:8501
2. Click "Model Comparison" → See diabetes accuracy (75.8%)
3. Click "Research Analysis" → Select "Cross-Validation Analysis"
4. Select "diabetes" → Click "Run Cross-Validation Analysis"
5. View detailed results with confidence intervals

### In Files:
1. Open `Multiple-Disease-Prediction-Webapp/Frontend/cv_results_diabetes.json`
2. See:
   - Mean accuracy: 0.7577
   - 95% CI: [0.7310, 0.7844]
   - All fold scores
   - Statistical tests

---

## 🔍 6. TROUBLESHOOTING

### Can't see results in application?
1. Make sure application is running (http://localhost:8501)
2. Navigate to correct section in sidebar
3. Click "Run Analysis" buttons if needed

### Can't find JSON files?
1. Check location: `Multiple-Disease-Prediction-Webapp/Frontend/`
2. Run cross-validation analysis in application first
3. Files are generated after running analysis

### SHAP plots not showing?
1. Go to Research Analysis
2. Select "SHAP XAI Analysis"
3. Select diseases
4. Click "Run SHAP Analysis"
5. Plots will be generated and displayed

---

## 📌 SUMMARY

### Best Way to See Results:
🌐 **Open the web application**: http://localhost:8501

### Quick Navigation:
1. **Model Comparison** → Overall performance
2. **Research Analysis** → Detailed analysis
3. **Disease Pages** → Individual predictions

### For Research Papers:
📁 **Use JSON files** in `Multiple-Disease-Prediction-Webapp/Frontend/`

---

**Everything is working and results are available both in the application and as files!** ✅
