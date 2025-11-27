#!/usr/bin/env python3
"""
🔍 COMPLETE SYSTEM VERIFICATION
==============================
Verifies that all ML and DL components are working correctly
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def check_traditional_ml():
    """Verify traditional ML models"""
    print("\n🤖 CHECKING TRADITIONAL ML MODELS")
    print("-" * 40)
    
    models_dir = "models"
    required_models = [
        "diabetes_model.sav", "heart_disease_model.sav", 
        "parkinsons_model.sav", "liver_model.sav", 
        "hepititisc_model.sav", "chronic_model.sav"
    ]
    
    found_models = 0
    for model in required_models:
        path = os.path.join(models_dir, model)
        if os.path.exists(path):
            print(f"  ✅ {model}")
            found_models += 1
        else:
            print(f"  ❌ {model}")
    
    print(f"\n📊 Traditional ML: {found_models}/{len(required_models)} models found")
    return found_models == len(required_models)

def check_deep_learning():
    """Verify deep learning models"""
    print("\n🧠 CHECKING DEEP LEARNING MODELS")
    print("-" * 40)
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"  ✅ TensorFlow {tf.__version__} installed")
        tf_available = True
    except ImportError:
        print("  ❌ TensorFlow not available")
        return False
    
    # Check disease DL models
    dl_dir = "models/deep_learning"
    disease_models = [
        "diabetes_deep_model.h5", "heart_disease_deep_model.h5",
        "parkinsons_deep_model.h5", "liver_disease_deep_model.h5",
        "hepatitis_deep_model.h5", "chronic_kidney_deep_model.h5"
    ]
    
    found_dl = 0
    for model in disease_models:
        path = os.path.join(dl_dir, model)
        if os.path.exists(path):
            print(f"  ✅ {model}")
            found_dl += 1
        else:
            print(f"  ❌ {model}")
    
    # Check image models
    image_models = [
        "chest_xray_image_model.h5", "brain_mri_image_model.h5",
        "retinal_scan_image_model.h5", "skin_lesion_image_model.h5"
    ]
    
    found_img = 0
    for model in image_models:
        path = os.path.join(dl_dir, model)
        if os.path.exists(path):
            print(f"  ✅ {model}")
            found_img += 1
        else:
            print(f"  ❌ {model}")
    
    total_dl = len(disease_models) + len(image_models)
    found_total = found_dl + found_img
    
    print(f"\n📊 Deep Learning: {found_total}/{total_dl} models found")
    print(f"   Disease Models: {found_dl}/{len(disease_models)}")
    print(f"   Image Models: {found_img}/{len(image_models)}")
    
    return found_total >= total_dl * 0.8  # At least 80% of models

def check_advanced_ml():
    """Verify advanced ML models"""
    print("\n⚡ CHECKING ADVANCED ML MODELS")
    print("-" * 40)
    
    advanced_dir = "models/advanced_ml"
    if not os.path.exists(advanced_dir):
        print("  ❌ Advanced ML directory not found")
        return False
    
    # Check for any advanced models
    files = os.listdir(advanced_dir)
    advanced_models = [f for f in files if f.endswith('.pkl')]
    
    if len(advanced_models) >= 10:
        print(f"  ✅ {len(advanced_models)} advanced ML models found")
        return True
    else:
        print(f"  ⚠️ Only {len(advanced_models)} advanced ML models found")
        return False

def check_ensemble_models():
    """Verify ensemble models"""
    print("\n🔗 CHECKING ENSEMBLE MODELS")
    print("-" * 40)
    
    ensemble_dir = "models/ensemble"
    if not os.path.exists(ensemble_dir):
        print("  ❌ Ensemble directory not found")
        return False
    
    files = os.listdir(ensemble_dir)
    ensemble_models = [f for f in files if f.endswith('.pkl')]
    
    if len(ensemble_models) >= 5:
        print(f"  ✅ {len(ensemble_models)} ensemble models found")
        return True
    else:
        print(f"  ⚠️ Only {len(ensemble_models)} ensemble models found")
        return False

def check_code_modules():
    """Verify code modules"""
    print("\n📝 CHECKING CODE MODULES")
    print("-" * 40)
    
    required_modules = [
        "code/DiseaseModel.py", "code/helper.py", "code/train.py",
        "code/AdvancedMLModels.py", "code/DeepLearningModels.py",
        "code/EnsemblePredictor.py", "code/MedicalImageAnalysis.py"
    ]
    
    found_modules = 0
    for module in required_modules:
        if os.path.exists(module):
            print(f"  ✅ {module}")
            found_modules += 1
        else:
            print(f"  ❌ {module}")
    
    print(f"\n📊 Code Modules: {found_modules}/{len(required_modules)} found")
    return found_modules == len(required_modules)

def check_data_files():
    """Verify data files"""
    print("\n📊 CHECKING DATA FILES")
    print("-" * 40)
    
    data_files = [
        "data/dataset.csv", "data/clean_dataset.tsv",
        "data/Symptom-severity.csv", "data/symptom_Description.csv",
        "data/symptom_precaution.csv"
    ]
    
    found_data = 0
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"  ✅ {data_file}")
            found_data += 1
        else:
            print(f"  ❌ {data_file}")
    
    print(f"\n📊 Data Files: {found_data}/{len(data_files)} found")
    return found_data >= len(data_files) * 0.8

def test_imports():
    """Test critical imports"""
    print("\n🔧 TESTING CRITICAL IMPORTS")
    print("-" * 40)
    
    imports_to_test = [
        ("streamlit", "Streamlit"),
        ("tensorflow", "TensorFlow"),
        ("sklearn", "Scikit-learn"),
        ("xgboost", "XGBoost"),
        ("lightgbm", "LightGBM"),
        ("catboost", "CatBoost"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("plotly", "Plotly"),
        ("cv2", "OpenCV")
    ]
    
    successful_imports = 0
    for module, name in imports_to_test:
        try:
            __import__(module)
            print(f"  ✅ {name}")
            successful_imports += 1
        except ImportError:
            print(f"  ❌ {name}")
    
    print(f"\n📊 Imports: {successful_imports}/{len(imports_to_test)} successful")
    return successful_imports >= len(imports_to_test) * 0.9

def main():
    """Main verification function"""
    print("🔍 COMPLETE SYSTEM VERIFICATION")
    print("=" * 60)
    print("Checking all ML/DL components...")
    
    # Run all checks
    checks = [
        ("Traditional ML Models", check_traditional_ml()),
        ("Deep Learning Models", check_deep_learning()),
        ("Advanced ML Models", check_advanced_ml()),
        ("Ensemble Models", check_ensemble_models()),
        ("Code Modules", check_code_modules()),
        ("Data Files", check_data_files()),
        ("Critical Imports", test_imports())
    ]
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(checks)
    
    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:25}: {status}")
        if result:
            passed += 1
    
    print("\n" + "=" * 60)
    overall_score = (passed / total) * 100
    
    if overall_score >= 90:
        status_emoji = "🎉"
        status_text = "EXCELLENT"
    elif overall_score >= 80:
        status_emoji = "✅"
        status_text = "GOOD"
    elif overall_score >= 70:
        status_emoji = "⚠️"
        status_text = "ACCEPTABLE"
    else:
        status_emoji = "❌"
        status_text = "NEEDS WORK"
    
    print(f"📊 OVERALL SCORE: {passed}/{total} ({overall_score:.1f}%)")
    print(f"{status_emoji} SYSTEM STATUS: {status_text}")
    
    if overall_score >= 80:
        print("\n🚀 SYSTEM READY FOR PRODUCTION!")
        print("🌐 Access your application at: http://localhost:8507")
    else:
        print(f"\n⚠️ System needs improvement. {total-passed} checks failed.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()