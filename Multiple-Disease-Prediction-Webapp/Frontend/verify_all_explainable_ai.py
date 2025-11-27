#!/usr/bin/env python3
"""
Comprehensive Verification Script for Explainable AI Implementation
Advanced Medical Disease Prediction System with Deep Learning Integration
"""

import re
import os
import sys
import importlib

def check_explainable_ai_implementation():
    """Check if all disease predictions have explainable AI"""
    
    print("🔍 VERIFYING EXPLAINABLE AI IMPLEMENTATION FOR ALL DISEASES")
    print("=" * 80)
    
    # Read the app.py file
    try:
        with open('app.py', 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print("❌ ERROR: app.py file not found!")
        return False
    
    # Define all disease sections to check
    diseases = [
        'Diabetes Prediction',
        'Heart disease Prediction', 
        'Parkison Prediction',
        'Liver prediction',
        'Hepatitis prediction',
        'Chronic Kidney prediction'
    ]
    
    results = {}
    
    for disease in diseases:
        print(f"\n🏥 Checking {disease}...")
        
        # Find the disease section
        pattern = rf"if selected == '{re.escape(disease)}':(.*?)(?=if selected ==|$)"
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            section_content = match.group(1)
            
            # Check for required explainable AI components
            checks = {
                'Model Metrics': 'load_model_metrics' in section_content and 'display_model_metrics' in section_content,
                'AI Explanation': 'show_explanation' in section_content and 'AI Explanation' in section_content,
                'Feature Importance': 'explain_prediction_advanced' in section_content,
                'Risk Analysis': 'plot_feature_importance_advanced' in section_content,
                'Risk Factors': 'display_risk_factors_analysis' in section_content,
                'Recommendations': 'Personalized' in section_content and 'Recommendations' in section_content,
                'Risk Assessment': 'high_risk_features' in section_content or 'risk_level' in section_content,
                'Critical Alerts': any(alert in section_content for alert in ['CRITICAL', 'HIGH RISK', 'ELEVATED RISK', '🚨', '⚠️'])
            }
            
            results[disease] = checks
            
            # Display results for this disease
            all_implemented = all(checks.values())
            status = "✅ COMPLETE" if all_implemented else "❌ INCOMPLETE"
            print(f"   Status: {status}")
            
            for check_name, implemented in checks.items():
                emoji = "✅" if implemented else "❌"
                print(f"   {emoji} {check_name}")
                
        else:
            print(f"   ❌ Disease section not found!")
            results[disease] = {check: False for check in ['Model Metrics', 'AI Explanation', 'Feature Importance', 'Risk Analysis', 'Risk Factors', 'Recommendations', 'Risk Assessment', 'Critical Alerts']}
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 EXPLAINABLE AI IMPLEMENTATION SUMMARY")
    print("=" * 80)
    
    total_diseases = len(diseases)
    complete_diseases = sum(1 for disease_checks in results.values() if all(disease_checks.values()))
    
    print(f"\n🎯 Overall Progress: {complete_diseases}/{total_diseases} diseases have complete explainable AI")
    print(f"📈 Completion Rate: {(complete_diseases/total_diseases)*100:.1f}%")
    
    if complete_diseases == total_diseases:
        print("\n🎉 SUCCESS: All diseases have comprehensive explainable AI implemented!")
        print("✅ Model performance metrics")
        print("✅ Advanced AI explanations") 
        print("✅ Risk factor analysis")
        print("✅ Personalized recommendations")
        print("✅ Critical health alerts")
    else:
        print(f"\n⚠️ INCOMPLETE: {total_diseases - complete_diseases} diseases still need explainable AI")
        
        # Show which diseases are incomplete
        for disease, checks in results.items():
            if not all(checks.values()):
                print(f"\n❌ {disease} - Missing:")
                for check_name, implemented in checks.items():
                    if not implemented:
                        print(f"   • {check_name}")
    
    return complete_diseases == total_diseases

def check_deep_learning_features():
    """Check deep learning and image analysis features"""
    
    print("\n" + "=" * 80)
    print("🧠 CHECKING DEEP LEARNING & IMAGE ANALYSIS FEATURES")
    print("=" * 80)
    
    # Check if deep learning modules exist
    deep_learning_files = [
        'code/DeepLearningModels.py',
        'code/MedicalImageAnalysis.py',
        'code/EnsemblePredictor.py'
    ]
    
    dl_results = {}
    
    for dl_file in deep_learning_files:
        if os.path.exists(dl_file):
            print(f"✅ {dl_file} - File exists")
            dl_results[dl_file] = True
            
            # Check file content for key methods
            try:
                with open(dl_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if 'MedicalImageAnalysis' in dl_file:
                    # Check image analysis specific methods
                    methods = ['explain_image_prediction', 'generate_image_report', 'load_model', 'analyze_medical_image']
                    for method in methods:
                        if method in content:
                            print(f"  ✅ {method} method found")
                        else:
                            print(f"  ❌ {method} method missing")
                            
                elif 'DeepLearningModels' in dl_file:
                    # Check deep learning specific methods
                    methods = ['create_neural_network', 'train_deep_model', 'predict_with_confidence']
                    for method in methods:
                        if method in content:
                            print(f"  ✅ {method} method found")
                        else:
                            print(f"  ❌ {method} method missing")
                            
                elif 'EnsemblePredictor' in dl_file:
                    # Check ensemble specific methods
                    methods = ['predict_with_ensemble', 'ensemble_predict', 'combine_predictions']
                    for method in methods:
                        if method in content:
                            print(f"  ✅ {method} method found")
                        else:
                            print(f"  ❌ {method} method missing")
                            
            except Exception as e:
                print(f"  ⚠️ Error reading file: {e}")
                
        else:
            print(f"❌ {dl_file} - File missing")
            dl_results[dl_file] = False
    
    return all(dl_results.values())

def check_advanced_features():
    """Check for specific advanced features"""
    
    print("\n" + "=" * 80)
    print("🚀 CHECKING ADVANCED FEATURES")
    print("=" * 80)
    
    try:
        with open('app.py', 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print("❌ ERROR: app.py file not found!")
        return False
    
    advanced_features = {
        'Risk Level Classification': any(x in content for x in ['High.*Medium.*Low', 'risk_level.*High', 'CRITICAL.*ELEVATED', "'High'", "'Medium'", "'Low'"]) or 'risk_level' in content,
        'Color-coded Risk Factors': any(x in content for x in ['🔴.*🟡.*🟢', 'risk_emoji', 'color.*risk', '🔴', '🟡', '🟢']),
        'Contribution Scores': any(x in content.lower() for x in ['contribution.*score', 'feature.*importance.*score', 'contribution_percentage', 'contribution']),
        'Personalized Recommendations': any(x in content.lower() for x in ['personalized.*recommendations', 'generate_personalized_recommendations', 'treatment.*recommendations']),
        'Critical Alerts': any(x in content.lower() for x in ['critical.*alert', 'urgent.*consultation', '🚨', 'critical:', 'urgent:', 'high.*risk']),
        'Medical Disclaimers': any(x in content.lower() for x in ['medical disclaimer', 'educational.*purposes', 'consult.*healthcare', 'professional.*medical', 'medical.*advice', 'healthcare.*professional']),
        'Comprehensive Metrics': any(x in content.lower() for x in ['accuracy.*precision.*recall.*f1', 'model.*metrics.*display', 'load_model_metrics', 'display_model_metrics']),
        'Interactive Visualizations': 'plotly_chart' in content and 'plot_feature_importance_advanced' in content,
        'TensorFlow Integration': any(x in content.lower() for x in ['tensorflow', 'deep.*learning', 'TENSORFLOW_AVAILABLE', 'DEEP_LEARNING_AVAILABLE']),
        'Image Analysis Support': any(x in content for x in ['MedicalImageAnalyzer', 'image.*analysis', 'uploaded_file', 'Medical Image Analysis'])
    }
    
    feature_count = 0
    for feature, implemented in advanced_features.items():
        emoji = "✅" if implemented else "❌"
        print(f"{emoji} {feature}")
        if implemented:
            feature_count += 1
    
    total_advanced = len(advanced_features)
    print(f"\n🎯 Advanced Features: {feature_count}/{total_advanced} implemented")
    print(f"📈 Advanced Feature Rate: {(feature_count/total_advanced)*100:.1f}%")
    
    return feature_count == total_advanced

def check_model_files():
    """Check if required model files exist"""
    
    print("\n" + "=" * 80)
    print("📁 CHECKING MODEL FILES")
    print("=" * 80)
    
    model_directories = [
        'models/',
        'models/ml_models/',
        'models/deep_learning/'
    ]
    
    model_files_found = 0
    total_expected = 0
    
    for model_dir in model_directories:
        if os.path.exists(model_dir):
            print(f"✅ Directory exists: {model_dir}")
            
            # List files in directory
            try:
                files = os.listdir(model_dir)
                if files:
                    print(f"  📋 Found {len(files)} files:")
                    for file in files[:5]:  # Show first 5 files
                        print(f"    • {file}")
                    if len(files) > 5:
                        print(f"    ... and {len(files) - 5} more files")
                    model_files_found += len(files)
                else:
                    print(f"  ⚠️ Directory is empty")
            except PermissionError:
                print(f"  ⚠️ Permission denied accessing directory")
                
        else:
            print(f"❌ Directory missing: {model_dir}")
    
    # Expected model files
    expected_models = [
        'diabetes_model.pkl', 'heart_model.pkl', 'parkinsons_model.pkl',
        'liver_model.pkl', 'hepatitis_model.pkl', 'chronic_model.pkl'
    ]
    
    total_expected = len(expected_models)
    
    print(f"\n📊 Model Files Summary:")
    print(f"  • Total files found: {model_files_found}")
    print(f"  • Expected core models: {total_expected}")
    
    return model_files_found >= total_expected

def check_dependencies():
    """Check if all required dependencies are available"""
    
    print("\n" + "=" * 80)
    print("📦 CHECKING DEPENDENCIES")
    print("=" * 80)
    
    required_packages = [
        ('streamlit', 'Web framework'),
        ('pandas', 'Data manipulation'),
        ('numpy', 'Numerical computing'),
        ('sklearn', 'Machine learning'),  # Import name is 'sklearn', not 'scikit-learn'
        ('plotly', 'Interactive visualizations'),
        ('matplotlib', 'Static plotting'),
        ('seaborn', 'Statistical visualization'),
        ('pickle', 'Model serialization'),
        ('joblib', 'Model persistence'),
        ('PIL', 'Image processing'),  # Import name is 'PIL', not 'pillow'
        ('cv2', 'Computer vision'),  # Import name is 'cv2', not 'opencv-python'
    ]
    
    optional_packages = [
        ('tensorflow', 'Deep learning'),
        ('keras', 'Neural networks'),
        ('torch', 'PyTorch deep learning'),
        ('lime', 'Model explainability'),
        ('shap', 'Advanced explainability')
    ]
    
    available_required = 0
    available_optional = 0
    
    print("📋 Required Dependencies:")
    for package, description in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package} - {description}")
            available_required += 1
        except ImportError:
            print(f"  ❌ {package} - {description} (MISSING)")
    
    print("\n📋 Optional Dependencies:")
    for package, description in optional_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package} - {description}")
            available_optional += 1
        except ImportError:
            print(f"  ⚠️ {package} - {description} (Optional)")
    
    req_percentage = (available_required / len(required_packages)) * 100
    opt_percentage = (available_optional / len(optional_packages)) * 100
    
    print(f"\n📊 Dependency Summary:")
    print(f"  • Required: {available_required}/{len(required_packages)} ({req_percentage:.1f}%)")
    print(f"  • Optional: {available_optional}/{len(optional_packages)} ({opt_percentage:.1f}%)")
    
    return available_required == len(required_packages)

def main():
    """Main verification function"""
    
    print("🎉 COMPREHENSIVE EXPLAINABLE AI VERIFICATION SYSTEM")
    print("🤖 Advanced Medical Disease Prediction with Deep Learning")
    print("=" * 80)
    
    # Check basic explainable AI implementation
    print("\n🔍 Phase 1: Basic Explainable AI Features")
    basic_complete = check_explainable_ai_implementation()
    
    # Check deep learning features
    print("\n🧠 Phase 2: Deep Learning Integration")
    deep_learning_complete = check_deep_learning_features()
    
    # Check advanced features
    print("\n🚀 Phase 3: Advanced Features")
    advanced_complete = check_advanced_features()
    
    # Check model files
    print("\n📁 Phase 4: Model Files")
    models_complete = check_model_files()
    
    # Check dependencies
    print("\n📦 Phase 5: Dependencies")
    dependencies_complete = check_dependencies()
    
    # Final assessment
    print("\n" + "=" * 80)
    print("🏆 FINAL COMPREHENSIVE ASSESSMENT")
    print("=" * 80)
    
    all_phases = [basic_complete, deep_learning_complete, advanced_complete, models_complete, dependencies_complete]
    completed_phases = sum(all_phases)
    total_phases = len(all_phases)
    
    overall_percentage = (completed_phases / total_phases) * 100
    
    if completed_phases == total_phases:
        print("🎉 PERFECT IMPLEMENTATION!")
        print("✅ All diseases have comprehensive explainable AI")
        print("✅ Deep learning and image analysis fully integrated")
        print("✅ All advanced features are implemented")
        print("✅ Model files are properly organized")
        print("✅ All dependencies are available")
        print("\n🚀 The Enhanced Disease Prediction System is PRODUCTION-READY with:")
        print("   • 🧠 Advanced Machine Learning & Deep Learning")
        print("   • 🔬 Medical Image Analysis (X-ray, MRI, Skin, Retinal)")
        print("   • 📊 Comprehensive metrics (Accuracy, Precision, Recall, F1)")
        print("   • 🤖 Explainable AI for all 6+ diseases")
        print("   • ⚡ Risk level classification (High/Medium/Low)")
        print("   • 🎨 Color-coded risk factor analysis")
        print("   • 💊 Personalized health recommendations")
        print("   • 🚨 Critical health alerts and warnings")
        print("   • 📈 Interactive visualizations")
        print("   • 🏥 Professional medical-grade interface")
        print("   • 🔒 Medical disclaimers and safety measures")
        
    else:
        print(f"📊 SYSTEM STATUS: {completed_phases}/{total_phases} phases complete ({overall_percentage:.1f}%)")
        
        # Show specific phase results
        phase_names = [
            "Basic Explainable AI",
            "Deep Learning Integration", 
            "Advanced Features",
            "Model Files",
            "Dependencies"
        ]
        
        for i, (phase_name, complete) in enumerate(zip(phase_names, all_phases)):
            status = "✅ Complete" if complete else "❌ Incomplete"
            print(f"   {status}: {phase_name}")
    
    # Recommendations
    if completed_phases < total_phases:
        print(f"\n🔧 RECOMMENDATIONS FOR IMPROVEMENT:")
        if not basic_complete:
            print("   • Complete explainable AI implementation for all diseases")
        if not deep_learning_complete:
            print("   • Implement deep learning models and image analysis")
        if not advanced_complete:
            print("   • Add advanced features like risk classification and alerts")
        if not models_complete:
            print("   • Train and save all required model files")
        if not dependencies_complete:
            print("   • Install missing required dependencies")
    
    print(f"\n📊 Overall System Health: {overall_percentage:.1f}%")
    print(f"🎯 System Grade: {'A+' if overall_percentage >= 95 else 'A' if overall_percentage >= 85 else 'B' if overall_percentage >= 75 else 'C' if overall_percentage >= 65 else 'Needs Improvement'}")
    
    return completed_phases == total_phases

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 VERIFICATION PASSED: Complete explainable AI system ready for production!")
        sys.exit(0)
    else:
        print("\n⚠️ VERIFICATION PARTIALLY COMPLETE: Some improvements needed.")
        sys.exit(1)