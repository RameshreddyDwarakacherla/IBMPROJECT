# 🏥 Multiple Disease Prediction Web Application

A comprehensive machine learning-based web application for predicting multiple diseases using Random Forest and SVM classifiers. Built with Streamlit and scikit-learn.

## 🌟 Features

### Disease Prediction Models
- **Diabetes Prediction** - Random Forest Classifier
- **Heart Disease Prediction** - Random Forest Classifier
- **Parkinson's Disease Prediction** - Support Vector Classifier (SVC)
- **Liver Disease Prediction** - Random Forest Classifier
- **Hepatitis Prediction** - Random Forest Classifier
- **Chronic Kidney Disease Prediction** - Random Forest Classifier

### Advanced Analysis Tools
- ✅ **Cross-Validation Analysis** - 10-fold CV with confidence intervals
- ✅ **SHAP Explainable AI** - Feature importance and interpretability (for tree-based models)
- ✅ **Hyperparameter Tuning** - Grid search optimization documentation
- ✅ **Model Comparison** - Performance metrics across all models
- ✅ **Advanced ML Models** - XGBoost, Gradient Boosting, Extra Trees

## 📊 Dataset Information

| Disease | Samples | Features | Model Type |
|---------|---------|----------|------------|
| Diabetes | 768 | 8 | Random Forest |
| Heart Disease | 303 | 13 | Random Forest |
| Parkinson's | 195 | 22 | SVC |
| Liver Disease | 583 | 10 | Random Forest |
| Hepatitis | 155 | 19 | Random Forest |
| Chronic Kidney | 400 | 24 | Random Forest |

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/RameshreddyDwarakacherla/IBMPROJECT.git
cd IBMPROJECT
```

2. **Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
cd Multiple-Disease-Prediction-Webapp/Frontend
streamlit run app.py
```

5. **Open in browser**
```
http://localhost:8501
```

## 📁 Project Structure

```
IBMPROJECT/
├── Multiple-Disease-Prediction-Webapp/
│   └── Frontend/
│       ├── app.py                          # Main Streamlit application
│       ├── models/                         # Trained ML models (.sav files)
│       │   ├── diabetes_model.sav
│       │   ├── heart_disease_model.sav
│       │   ├── parkinsons_model.sav
│       │   ├── liver_model.sav
│       │   ├── hepititisc_model.sav
│       │   └── chronic_model.sav
│       └── data/                           # Training datasets (.csv files)
│           ├── diabetes.csv
│           ├── heart.csv
│           ├── parkinsons.csv
│           ├── indian_liver_patient.csv
│           ├── hepatitis.csv
│           └── kidney_disease.csv
├── cross_validation_analysis.py           # CV analysis script
├── hyperparameter_tuning_analysis.py      # Hyperparameter tuning script
├── shap_xai_analysis.py                   # SHAP analysis script
├── paper_revision_checklist.md            # Research paper revision guide
├── comparison_tables_for_paper.md         # Performance comparison tables
├── reviewer_response_letter.md            # Response to reviewers
├── revised_introduction.md                # Revised paper introduction
└── README.md                              # This file
```

## 🎯 Usage

### 1. Disease Prediction
- Select a disease from the sidebar
- Enter patient parameters
- Click "Predict" to get results
- View prediction confidence and risk assessment

### 2. Research Analysis Tools
Navigate to "Research Analysis" in the sidebar:

#### Cross-Validation Analysis
- Select diseases to analyze
- Run 10-fold cross-validation
- View mean accuracy with 95% confidence intervals
- Generate statistical reports

#### SHAP Explainable AI
- Available for: Diabetes, Heart, Liver (tree-based models)
- View SHAP summary plots
- Analyze feature importance
- Examine dependence plots

#### Hyperparameter Tuning
- Document hyperparameter search spaces
- View optimal parameters
- Compare performance improvements

#### Model Comparison
- Compare all disease models
- View accuracy, precision, recall, F1-score
- Analyze ROC curves and confusion matrices

## 📈 Model Performance

### Accuracy Scores

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Diabetes | 98.70% | 0.99 | 0.98 | 0.98 |
| Heart Disease | 85.25% | 0.86 | 0.85 | 0.85 |
| Parkinson's | 89.74% | 0.90 | 0.90 | 0.90 |
| Liver Disease | 71.55% | 0.72 | 0.72 | 0.71 |
| Hepatitis | 83.87% | 0.84 | 0.84 | 0.84 |
| Chronic Kidney | 99.00% | 0.99 | 0.99 | 0.99 |

## 🔬 Research Paper Support

This project includes comprehensive documentation for research paper preparation:

- **paper_revision_checklist.md** - Complete revision checklist
- **comparison_tables_for_paper.md** - Performance comparison tables
- **reviewer_response_letter.md** - Template for responding to reviewers
- **revised_introduction.md** - Updated paper introduction
- **QUICK_START_GUIDE.md** - Quick reference for paper content

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **ML Libraries**: scikit-learn, XGBoost
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Explainability**: SHAP
- **Model Persistence**: joblib

## 📦 Dependencies

```
streamlit
scikit-learn
pandas
numpy
matplotlib
seaborn
plotly
xgboost
shap
joblib
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Ramesh Reddy Dwarakacherla** - [GitHub](https://github.com/RameshreddyDwarakacherla)

## 🙏 Acknowledgments

- Dataset sources: UCI Machine Learning Repository, Kaggle
- SHAP library for explainable AI
- Streamlit for the web framework
- scikit-learn for machine learning algorithms

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🔗 Links

- **GitHub Repository**: https://github.com/RameshreddyDwarakacherla/IBMPROJECT
- **Live Demo**: [Coming Soon]

---

**Note**: This application is for educational and research purposes only. Always consult healthcare professionals for medical advice.
