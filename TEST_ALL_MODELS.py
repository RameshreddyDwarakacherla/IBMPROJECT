"""
Comprehensive test script to check all disease prediction models
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd

print("=" * 70)
print("TESTING ALL DISEASE PREDICTION MODELS")
print("=" * 70)

# Change to Frontend directory
os.chdir('Multiple-Disease-Prediction-Webapp/Frontend')

# Test data for each disease
test_cases = {
    'diabetes': {
        'model_file': 'models/diabetes_model.sav',
        'test_data': np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]]),
        'features': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    },
    'heart_disease': {
        'model_file': 'models/heart_disease_model.sav',
        'test_data': np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]),
        'features': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    },
    'parkinsons': {
        'model_file': 'models/parkinsons_model.sav',
        'test_data': np.array([[119.992, 157.302, 74.997, 0.00784, 0.00007, 0.00370,
                               0.00554, 0.01109, 0.04374, 0.426, 0.02182, 0.03130,
                               0.02971, 0.06545, 0.02211, 21.033, 0.414783, 0.815285,
                               -4.813031, 0.266482, 2.301442, 0.284654]]),
        'features': ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
                    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
                    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
                    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
                    'spread1', 'spread2', 'D2', 'PPE']
    },
    'liver': {
        'model_file': 'models/liver_model.sav',
        'test_data': np.array([[65, 0, 0.7, 0.1, 187, 16, 18, 6.8, 3.3, 0.9]]),
        'features': ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
                    'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
                    'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
                    'Albumin_and_Globulin_Ratio']
    },
    'hepatitis': {
        'model_file': 'models/hepititisc_model.sav',
        'test_data': np.array([[1, 30, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 85, 18, 4, 0, 0, 0]]),
        'features': ['ID', 'Age', 'Gender', 'Steroid', 'Antivirals', 'Fatigue', 'Malaise',
                    'Anorexia', 'LiverBig', 'LiverFirm', 'Spleen', 'Spiders',
                    'Ascites', 'Varices', 'Bili', 'Alk', 'Sgot',
                    'Albu', 'Protime', 'Histology']
    },
    'chronic_kidney': {
        'model_file': 'models/chronic_model.sav',
        'test_data': np.array([[48, 80, 1.020, 1, 0, 0, 0, 0, 0, 121, 36, 1.2, 137, 4.5,
                               15.4, 44, 7800, 5.2, 0, 0, 0, 1, 0, 0]]),
        'features': ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr',
                    'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm',
                    'cad', 'appet', 'pe', 'ane']
    }
}

results = {}
errors = []

for disease, config in test_cases.items():
    print(f"\n{'='*70}")
    print(f"Testing {disease.upper()} Model")
    print(f"{'='*70}")
    
    try:
        # Check if model file exists
        if not os.path.exists(config['model_file']):
            error_msg = f"Model file not found: {config['model_file']}"
            print(f"❌ {error_msg}")
            errors.append((disease, error_msg))
            results[disease] = 'MISSING'
            continue
        
        # Load the model
        print(f"Loading model from: {config['model_file']}")
        model = joblib.load(config['model_file'])
        print(f"✓ Model loaded successfully")
        print(f"  Model type: {type(model).__name__}")
        
        # Test prediction
        print(f"\nTesting prediction with sample data...")
        print(f"  Input shape: {config['test_data'].shape}")
        print(f"  Features: {len(config['features'])}")
        
        prediction = model.predict(config['test_data'])
        print(f"✓ Prediction successful: {prediction[0]}")
        
        # Test probability prediction if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(config['test_data'])
            print(f"✓ Probability prediction: {proba[0]}")
        
        # Test feature importance if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            print(f"✓ Feature importances available: {len(importances)} features")
            top_3 = sorted(zip(config['features'], importances), 
                          key=lambda x: x[1], reverse=True)[:3]
            print(f"  Top 3 features:")
            for feat, imp in top_3:
                print(f"    - {feat}: {imp:.4f}")
        
        results[disease] = 'WORKING'
        print(f"\n✅ {disease.upper()} model is WORKING correctly!")
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Error testing {disease}: {error_msg}")
        errors.append((disease, error_msg))
        results[disease] = 'ERROR'
        import traceback
        traceback.print_exc()

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

working = sum(1 for v in results.values() if v == 'WORKING')
missing = sum(1 for v in results.values() if v == 'MISSING')
error = sum(1 for v in results.values() if v == 'ERROR')

print(f"\nTotal Models: {len(test_cases)}")
print(f"✅ Working: {working}")
print(f"❌ Missing: {missing}")
print(f"⚠️  Errors: {error}")

print(f"\nDetailed Results:")
for disease, status in results.items():
    icon = '✅' if status == 'WORKING' else '❌' if status == 'MISSING' else '⚠️'
    print(f"  {icon} {disease.ljust(20)}: {status}")

if errors:
    print(f"\n{'='*70}")
    print("ERRORS TO FIX")
    print(f"{'='*70}")
    for disease, error in errors:
        print(f"\n{disease.upper()}:")
        print(f"  {error}")

print(f"\n{'='*70}")
if working == len(test_cases):
    print("✅ ALL MODELS ARE WORKING!")
else:
    print(f"⚠️  {len(test_cases) - working} MODEL(S) NEED ATTENTION")
print(f"{'='*70}\n")
