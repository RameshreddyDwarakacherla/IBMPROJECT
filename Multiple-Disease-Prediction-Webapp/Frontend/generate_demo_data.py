#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate demonstration data for cross-validation analysis
This creates sample CSV files when original training data is not available
"""

import pandas as pd
import numpy as np
import os
import sys

# Ensure UTF-8 encoding
if sys.version_info[0] >= 3:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def generate_diabetes_data(n_samples=768):
    """Generate synthetic diabetes data similar to Pima Indians dataset"""
    np.random.seed(42)
    
    data = {
        'Pregnancies': np.random.randint(0, 17, n_samples),
        'Glucose': np.random.randint(0, 200, n_samples),
        'BloodPressure': np.random.randint(0, 122, n_samples),
        'SkinThickness': np.random.randint(0, 99, n_samples),
        'Insulin': np.random.randint(0, 846, n_samples),
        'BMI': np.random.uniform(0, 67.1, n_samples),
        'DiabetesPedigreeFunction': np.random.uniform(0.078, 2.42, n_samples),
        'Age': np.random.randint(21, 81, n_samples),
        'Outcome': np.random.randint(0, 2, n_samples)
    }
    
    return pd.DataFrame(data)

def generate_heart_data(n_samples=303):
    """Generate synthetic heart disease data"""
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(29, 77, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.randint(94, 200, n_samples),
        'chol': np.random.randint(126, 564, n_samples),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.randint(71, 202, n_samples),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.random.uniform(0, 6.2, n_samples),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.randint(0, 4, n_samples),
        'target': np.random.randint(0, 2, n_samples)
    }
    
    return pd.DataFrame(data)

def generate_parkinsons_data(n_samples=195):
    """Generate synthetic Parkinson's data"""
    np.random.seed(42)
    
    data = {
        'MDVP:Fo(Hz)': np.random.uniform(88, 260, n_samples),
        'MDVP:Fhi(Hz)': np.random.uniform(102, 592, n_samples),
        'MDVP:Flo(Hz)': np.random.uniform(65, 239, n_samples),
        'MDVP:Jitter(%)': np.random.uniform(0.00168, 0.03316, n_samples),
        'MDVP:Jitter(Abs)': np.random.uniform(0.000007, 0.000260, n_samples),
        'MDVP:RAP': np.random.uniform(0.00068, 0.02144, n_samples),
        'MDVP:PPQ': np.random.uniform(0.00092, 0.01958, n_samples),
        'Jitter:DDP': np.random.uniform(0.00204, 0.06433, n_samples),
        'MDVP:Shimmer': np.random.uniform(0.00954, 0.11908, n_samples),
        'MDVP:Shimmer(dB)': np.random.uniform(0.085, 1.302, n_samples),
        'Shimmer:APQ3': np.random.uniform(0.00455, 0.05647, n_samples),
        'Shimmer:APQ5': np.random.uniform(0.0057, 0.0794, n_samples),
        'MDVP:APQ': np.random.uniform(0.00719, 0.13778, n_samples),
        'Shimmer:DDA': np.random.uniform(0.01364, 0.16942, n_samples),
        'NHR': np.random.uniform(0.00065, 0.31482, n_samples),
        'HNR': np.random.uniform(8.441, 33.047, n_samples),
        'RPDE': np.random.uniform(0.256570, 0.685151, n_samples),
        'DFA': np.random.uniform(0.574282, 0.825288, n_samples),
        'spread1': np.random.uniform(-7.964984, -2.434031, n_samples),
        'spread2': np.random.uniform(0.006274, 0.450493, n_samples),
        'D2': np.random.uniform(1.423287, 3.671155, n_samples),
        'PPE': np.random.uniform(0.044539, 0.527367, n_samples),
        'status': np.random.randint(0, 2, n_samples)
    }
    
    return pd.DataFrame(data)

def main():
    """Generate all demo data files"""
    print("🔬 Generating demonstration data for cross-validation...")
    
    os.makedirs('data', exist_ok=True)
    
    # Generate datasets
    datasets = {
        'diabetes.csv': generate_diabetes_data(),
        'heart.csv': generate_heart_data(),
        'parkinsons.csv': generate_parkinsons_data()
    }
    
    for filename, df in datasets.items():
        filepath = os.path.join('data', filename)
        df.to_csv(filepath, index=False)
        print(f"✅ Created {filepath} ({len(df)} samples, {len(df.columns)} features)")
    
    print("\n✅ Demo data generation complete!")
    print("📁 Files created in data/ folder")
    print("🔬 You can now run cross-validation analysis")

if __name__ == "__main__":
    main()
