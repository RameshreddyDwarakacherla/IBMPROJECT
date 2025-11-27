#!/usr/bin/env python3
"""
Cross-Validation and Statistical Analysis for Paper Revision
Addresses Reviewer Comment: "Report confidence intervals or k-fold validation"
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
import json
from scipy import stats

class CrossValidationAnalyzer:
    """Perform comprehensive cross-validation analysis for all disease models"""
    
    def __init__(self, n_folds=10, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.results = {}
        
    def load_disease_data(self, disease_name):
        """Load dataset for specific disease"""
        data_paths = {
            'diabetes': 'Multiple-Disease-Prediction-Webapp/Frontend/data/diabetes.csv',
            'heart': 'Multiple-Disease-Prediction-Webapp/Frontend/data/heart.csv',
            'parkinsons': 'Multiple-Disease-Prediction-Webapp/Frontend/data/parkinsons.csv',
            'liver': 'Multiple-Disease-Prediction-Webapp/Frontend/data/indian_liver_patient.csv',
            'hepatitis': 'Multiple-Disease-Prediction-Webapp/Frontend/data/hepatitis.csv',
            'kidney': 'Multiple-Disease-Prediction-Webapp/Frontend/data/kidney_disease.csv',
            'lung_cancer': 'Multiple-Disease-Prediction-Webapp/Frontend/data/lung_cancer.csv'
        }
        
        try:
            df = pd.read_csv(data_paths[disease_name])
            
            # Handle different label encodings
            if disease_name == 'lung_cancer':
                # Convert YES/NO to 1/0
                if df.iloc[:, -1].dtype == 'object':
                    df.iloc[:, -1] = df.iloc[:, -1].map({'YES': 1, 'NO': 0})
                # Convert M/F to 1/0 for GENDER column
                if 'GENDER' in df.columns:
                    df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})
            
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            return X, y, df.columns[:-1].tolist()
        except Exception as e:
            print(f"Error loading {disease_name} data: {e}")
            return None, None, None
    
    def perform_cross_validation(self, model, X, y, model_name, disease_name):
        """Perform k-fold cross-validation with multiple metrics"""
        
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted',
            'f1': 'f1_weighted'
        }
        
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1)
        
        results = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            test_scores = cv_results[f'test_{metric}']
            mean_test = np.mean(test_scores)
            std_test = np.std(test_scores)
            ci_95 = 1.96 * (std_test / np.sqrt(self.n_folds))
            
            results[metric] = {
                'mean': mean_test,
                'std': std_test,
                'ci_95_lower': mean_test - ci_95,
                'ci_95_upper': mean_test + ci_95,
                'fold_scores': test_scores.tolist()
            }
        
        return results
    
    def compare_models(self, X, y, disease_name):
        """Compare RF, XGBoost, and SVM with statistical tests"""
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state, class_weight='balanced'),
            'XGBoost': XGBClassifier(n_estimators=100, random_state=self.random_state, use_label_encoder=False, eval_metric='logloss'),
            'SVM': SVC(kernel='rbf', random_state=self.random_state, class_weight='balanced')
        }
        
        comparison_results = {}
        accuracy_scores = {}
        
        for model_name, model in models.items():
            print(f"  Evaluating {model_name}...")
            results = self.perform_cross_validation(model, X, y, model_name, disease_name)
            comparison_results[model_name] = results
            accuracy_scores[model_name] = results['accuracy']['fold_scores']
        
        # ANOVA test
        f_stat, p_value = stats.f_oneway(*accuracy_scores.values())
        comparison_results['statistical_tests'] = {
            'anova': {'f_statistic': f_stat, 'p_value': p_value, 'significant': p_value < 0.05}
        }
        
        return comparison_results
    
    def analyze_all_diseases(self):
        """Analyze all 7 diseases"""
        diseases = ['diabetes', 'heart', 'parkinsons', 'liver', 'hepatitis', 'kidney', 'lung_cancer']
        
        for disease in diseases:
            print(f"\nAnalyzing {disease.upper()}")
            X, y, features = self.load_disease_data(disease)
            
            if X is not None:
                results = self.compare_models(X, y, disease)
                self.results[disease] = results
                self.save_results(disease, results)
    
    def save_results(self, disease_name, results):
        """Save results to JSON file"""
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list): return [convert(item) for item in obj]
            return obj
        
        with open(f"cv_results_{disease_name}.json", 'w') as f:
            json.dump(convert(results), f, indent=2)
        print(f"  ✅ Results saved")

def main():
    print("🔬 CROSS-VALIDATION ANALYSIS FOR PAPER REVISION")
    analyzer = CrossValidationAnalyzer(n_folds=10, random_state=42)
    analyzer.analyze_all_diseases()
    print("\n✅ ANALYSIS COMPLETE!")

if __name__ == "__main__":
    main()
