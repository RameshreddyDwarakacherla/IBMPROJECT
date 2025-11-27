#!/usr/bin/env python3
"""
SHAP (SHapley Additive exPlanations) Analysis for XAI
Addresses Reviewer Comment: "Specify and visualize the explainability mechanism used"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import joblib
import shap

class SHAPAnalyzer:
    """Generate SHAP explanations for disease prediction models"""
    
    def __init__(self):
        self.explainers = {}
        self.shap_values = {}
        
    def load_model_and_data(self, disease_name):
        """Load trained model and dataset"""
        model_paths = {
            'diabetes': 'Multiple-Disease-Prediction-Webapp/Frontend/models/diabetes_model.pkl',
            'heart': 'Multiple-Disease-Prediction-Webapp/Frontend/models/heart_model.pkl',
            'liver': 'Multiple-Disease-Prediction-Webapp/Frontend/models/liver_model.pkl'
        }
        
        data_paths = {
            'diabetes': 'Multiple-Disease-Prediction-Webapp/Frontend/data/diabetes.csv',
            'heart': 'Multiple-Disease-Prediction-Webapp/Frontend/data/heart.csv',
            'liver': 'Multiple-Disease-Prediction-Webapp/Frontend/data/indian_liver_patient.csv'
        }
        
        try:
            model = joblib.load(model_paths[disease_name])
            df = pd.read_csv(data_paths[disease_name])
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            return model, X, y, df.columns[:-1].tolist()
        except Exception as e:
            print(f"Error loading {disease_name}: {e}")
            return None, None, None, None
    
    def generate_shap_explanations(self, disease_name):
        """Generate SHAP values for a disease model"""
        print(f"\n🔍 Generating SHAP explanations for {disease_name.upper()}")
        
        model, X, y, feature_names = self.load_model_and_data(disease_name)
        if model is None:
            return
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # For binary classification, get positive class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        self.explainers[disease_name] = explainer
        self.shap_values[disease_name] = {
            'values': shap_values,
            'data': X,
            'feature_names': feature_names
        }
        
        print(f"  ✅ SHAP values computed")
        
        # Generate visualizations
        self.plot_shap_summary(disease_name)
        self.plot_shap_importance(disease_name)
        self.plot_shap_dependence(disease_name)
    
    def plot_shap_summary(self, disease_name):
        """Create SHAP summary plot"""
        data = self.shap_values[disease_name]
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            data['values'], 
            data['data'], 
            feature_names=data['feature_names'],
            show=False
        )
        plt.title(f'SHAP Summary Plot - {disease_name.capitalize()}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save in current directory (where Streamlit is running)
        import os
        filename = f'shap_summary_{disease_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print absolute path for debugging
        abs_path = os.path.abspath(filename)
        print(f"  ✅ Summary plot saved: {abs_path}")
    
    def plot_shap_importance(self, disease_name):
        """Create SHAP feature importance bar plot"""
        data = self.shap_values[disease_name]
        
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(data['values']).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': data['feature_names'],
            'importance': mean_shap
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
        plt.xlabel('Mean |SHAP value|', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'SHAP Feature Importance - {disease_name.capitalize()}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save in current directory (where Streamlit is running)
        import os
        filename = f'shap_importance_{disease_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print absolute path for debugging
        abs_path = os.path.abspath(filename)
        print(f"  ✅ Importance plot saved: {abs_path}")
    
    def plot_shap_dependence(self, disease_name):
        """Create SHAP dependence plots for top 3 features"""
        data = self.shap_values[disease_name]
        
        # Get top 3 features
        mean_shap = np.abs(data['values']).mean(axis=0)
        top_features_idx = np.argsort(mean_shap)[-3:]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, feature_idx in enumerate(top_features_idx):
            feature_name = data['feature_names'][feature_idx]
            
            shap.dependence_plot(
                feature_idx,
                data['values'],
                data['data'],
                feature_names=data['feature_names'],
                ax=axes[idx],
                show=False
            )
            axes[idx].set_title(f'{feature_name}', fontsize=12, fontweight='bold')
        
        plt.suptitle(f'SHAP Dependence Plots - {disease_name.capitalize()}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save in current directory (where Streamlit is running)
        import os
        filename = f'shap_dependence_{disease_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print absolute path for debugging
        abs_path = os.path.abspath(filename)
        print(f"  ✅ Dependence plots saved: {abs_path}")
    
    def generate_latex_xai_section(self):
        """Generate LaTeX section for XAI methodology"""
        latex = """
\\subsection{Explainable AI Framework}

To ensure transparency and clinical interpretability, we implemented a comprehensive 
Explainable AI (XAI) framework using SHAP (SHapley Additive exPlanations) \\cite{lundberg2017unified}. 
SHAP values provide a unified measure of feature importance based on game theory, 
offering both global and local interpretability.

\\subsubsection{SHAP Methodology}

For each prediction, SHAP computes the contribution of each feature by calculating 
Shapley values from cooperative game theory:

\\begin{equation}
\\phi_i = \\sum_{S \\subseteq F \\setminus \\{i\\}} \\frac{|S|!(|F|-|S|-1)!}{|F|!} [f_{S \\cup \\{i\\}}(x_{S \\cup \\{i\\}}) - f_S(x_S)]
\\end{equation}

where $\\phi_i$ is the SHAP value for feature $i$, $F$ is the set of all features, 
$S$ is a subset of features, and $f$ is the model prediction function.

\\subsubsection{Risk Classification}

Based on SHAP values and feature contributions, we classify risk factors into three categories:

\\begin{itemize}
    \\item \\textbf{High Risk (🔴):} Features with contribution $\\geq$ 75th percentile
    \\item \\textbf{Medium Risk (🟡):} Features with contribution between 50th-75th percentile
    \\item \\textbf{Low Risk (🟢):} Features with contribution $<$ 50th percentile
\\end{itemize}

\\subsubsection{Visualization and Interpretation}

We provide three types of SHAP visualizations:

\\begin{enumerate}
    \\item \\textbf{Summary Plots:} Show the distribution of SHAP values for all features
    \\item \\textbf{Feature Importance:} Rank features by mean absolute SHAP value
    \\item \\textbf{Dependence Plots:} Illustrate the relationship between feature values and SHAP values
\\end{enumerate}

This multi-faceted approach ensures that healthcare professionals can understand 
not only \\textit{what} the model predicts, but \\textit{why} it makes specific predictions, 
enabling informed clinical decision-making.
"""
        
        with open('xai_methodology_section.tex', 'w') as f:
            f.write(latex)
        
        print("\n✅ LaTeX XAI section saved: xai_methodology_section.tex")
    
    def analyze_all_diseases(self):
        """Analyze top 3 diseases with SHAP"""
        diseases = ['diabetes', 'heart', 'liver']
        
        for disease in diseases:
            self.generate_shap_explanations(disease)
        
        self.generate_latex_xai_section()

def main():
    print("🧠 SHAP EXPLAINABLE AI ANALYSIS")
    print("="*60)
    print("Addresses reviewer comment:")
    print("'Specify and visualize the explainability mechanism used (e.g., SHAP output)'")
    print("="*60)
    
    # Check if SHAP is installed
    try:
        import shap
        print("✅ SHAP library available")
    except ImportError:
        print("❌ SHAP not installed. Install with: pip install shap")
        return
    
    analyzer = SHAPAnalyzer()
    analyzer.analyze_all_diseases()
    
    print("\n✅ SHAP ANALYSIS COMPLETE!")
    print("\nGenerated files:")
    print("  • shap_summary_*.png - Summary plots for each disease")
    print("  • shap_importance_*.png - Feature importance plots")
    print("  • shap_dependence_*.png - Dependence plots for top features")
    print("  • xai_methodology_section.tex - LaTeX section for paper")
    print("\n📝 Add these visualizations to your revised paper!")

if __name__ == "__main__":
    main()
