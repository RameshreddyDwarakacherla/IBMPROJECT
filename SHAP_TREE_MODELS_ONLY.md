# ✅ SHAP Analysis - Tree-Based Models Only

## Summary

SHAP analysis now only includes diseases with **tree-based models** (Random Forest):
- ✅ **Diabetes** - RandomForestClassifier
- ✅ **Heart** - RandomForestClassifier  
- ✅ **Liver** - RandomForestClassifier
- ❌ **Parkinson's** - SVC (removed from SHAP)

## Why Parkinson's Was Removed

**Model Type Issue:**
- Parkinson's uses **SVC (Support Vector Classifier)**
- SHAP's `TreeExplainer` only works with tree-based models
- Using wrong explainer causes: `InvalidModelError: Model type not yet supported by TreeExplainer`

**Alternative Explainers:**
- `KernelExplainer` - Works with any model but very slow (hours for large datasets)
- `LinearExplainer` - Only for linear models
- `DeepExplainer` - Only for neural networks

For research purposes, it's better to focus on the three tree-based models that work efficiently.

## Available Models

| Disease | Model Type | SHAP Support | Status |
|---------|-----------|--------------|--------|
| Diabetes | RandomForestClassifier | ✅ TreeExplainer | Available |
| Heart | RandomForestClassifier | ✅ TreeExplainer | Available |
| Liver | RandomForestClassifier | ✅ TreeExplainer | Available |
| Parkinson's | SVC | ❌ Not compatible | Removed |

## What Changed

### 1. Updated Disease Selection
```python
# Before
["diabetes", "heart", "liver", "parkinsons"]

# After (tree-based only)
["diabetes", "heart", "liver"]
```

### 2. Removed Parkinson's Preprocessing
- Removed special handling for 'status' column
- Removed 'name' column dropping logic
- Simplified code to handle only standard datasets

### 3. Added Info Message
```
ℹ️ SHAP analysis works with tree-based models (Random Forest). 
Currently available: Diabetes, Heart, Liver
```

## How to Use

### 1. Start the App
```bash
cd Multiple-Disease-Prediction-Webapp\Frontend
streamlit run app.py
```

### 2. Navigate to SHAP
- Click **Research Analysis** in sidebar
- Select **SHAP XAI Analysis**

### 3. Select Diseases
Choose from:
- ✅ Diabetes (768 samples, 8 features)
- ✅ Heart (303 samples, 13 features)
- ✅ Liver (583 samples, 10 features)

### 4. Run Analysis
- Click **🚀 Run SHAP Analysis**
- Wait 1-2 minutes per disease
- View three types of plots for each

## What You'll See

For each disease:

### 1. SHAP Summary Plot
- Beeswarm plot showing all features
- Color = feature value (red=high, blue=low)
- Position = impact on prediction

### 2. Feature Importance
- Bar chart ranking features by importance
- Based on mean |SHAP value|
- Shows which features matter most

### 3. Dependence Plots
- Three plots for top 3 features
- Shows feature value vs SHAP value relationship
- Reveals interaction effects

## Performance

| Disease | Samples | Features | Analysis Time |
|---------|---------|----------|---------------|
| Diabetes | 768 | 8 | ~1-2 minutes |
| Heart | 303 | 13 | ~30-60 seconds |
| Liver | 583 | 10 | ~1 minute |

## For Your Paper

### What to Write

> "We applied SHAP (SHapley Additive exPlanations) analysis to three disease prediction models using Random Forest classifiers: diabetes, heart disease, and liver disease. SHAP values were computed using TreeExplainer, providing both global feature importance rankings and local explanations for individual predictions."

### Why Only Three Diseases

> "SHAP analysis was performed on models using tree-based algorithms (Random Forest), which are compatible with SHAP's efficient TreeExplainer. The Parkinson's disease model uses Support Vector Classification (SVC), which would require KernelExplainer—a model-agnostic approach that is computationally expensive for large-scale analysis."

### Key Findings to Include

1. **Feature Importance Rankings** - Which features are most important for each disease
2. **Feature Interactions** - How features interact (shown in dependence plots)
3. **Directional Effects** - Whether high/low feature values increase/decrease risk
4. **Model Transparency** - SHAP provides interpretable explanations for predictions

## Benefits of This Approach

✅ **Fast and Efficient** - TreeExplainer is optimized for tree-based models
✅ **Accurate** - Exact SHAP values, not approximations
✅ **Consistent** - All three diseases use same model type (Random Forest)
✅ **Reliable** - No errors or compatibility issues
✅ **Comprehensive** - Three diseases provide sufficient evidence of explainability

## Alternative for Parkinson's

If you need explainability for Parkinson's model:

### Option 1: Use Permutation Importance
```python
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X, y, n_repeats=10)
```

### Option 2: Retrain with Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
# Now compatible with SHAP TreeExplainer
```

### Option 3: Use KernelExplainer (Slow)
```python
explainer = shap.KernelExplainer(model.predict_proba, X_sample)
shap_values = explainer.shap_values(X_test)
# Warning: Can take hours for large datasets
```

## Troubleshooting

### If you still see Parkinson's option:
- Restart the Streamlit app
- Clear browser cache (Ctrl+F5)

### If SHAP fails for other diseases:
- Check model files exist: `dir models\*.sav`
- Check data files exist: `dir data\*.csv`
- Verify SHAP installed: `pip install shap`

### If analysis is slow:
- Normal for large datasets
- Diabetes takes longest (768 samples)
- Heart is fastest (303 samples)

## Summary

**Before:**
- 4 diseases (diabetes, heart, liver, parkinsons)
- Parkinson's caused errors (SVC not compatible)

**After:**
- 3 diseases (diabetes, heart, liver)
- All use Random Forest (tree-based)
- All work perfectly with SHAP TreeExplainer
- Fast, accurate, and reliable

---

**Status**: ✅ **OPTIMIZED FOR TREE-BASED MODELS**

SHAP now only includes compatible models - no more errors!
