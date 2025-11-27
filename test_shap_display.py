#!/usr/bin/env python3
"""
Quick test to verify SHAP analysis generates and saves images correctly
"""

import os
import sys

print("🧪 Testing SHAP Display Fix...")
print("="*60)

# Change to Frontend directory (where Streamlit runs)
original_dir = os.getcwd()
frontend_dir = os.path.join(original_dir, 'Multiple-Disease-Prediction-Webapp', 'Frontend')

if os.path.exists(frontend_dir):
    os.chdir(frontend_dir)
    print(f"✅ Changed to Frontend directory")
    print(f"📁 Working directory: {os.getcwd()}")
else:
    print(f"⚠️  Frontend directory not found, staying in: {os.getcwd()}")

# Test 1: Check if shap_xai_analysis.py exists
sys.path.append(original_dir)
if os.path.exists(os.path.join(original_dir, 'shap_xai_analysis.py')):
    print("✅ shap_xai_analysis.py found")
else:
    print("❌ shap_xai_analysis.py not found")
    sys.exit(1)

# Test 2: Check if SHAP library is available
try:
    import shap
    print("✅ SHAP library installed")
except ImportError:
    print("❌ SHAP not installed. Run: pip install shap")
    sys.exit(1)

# Test 3: Try to import and run analyzer
try:
    from shap_xai_analysis import SHAPAnalyzer
    print("✅ SHAPAnalyzer imported successfully")
    
    analyzer = SHAPAnalyzer()
    print("✅ Analyzer initialized")
    
    # Test with diabetes only (fastest)
    print("\n🔬 Running SHAP analysis for diabetes...")
    analyzer.generate_shap_explanations('diabetes')
    
    # Check if files were created in current directory
    expected_files = [
        'shap_summary_diabetes.png',
        'shap_importance_diabetes.png',
        'shap_dependence_diabetes.png'
    ]
    
    print("\n📁 Checking generated files in current directory:")
    found_count = 0
    for file in expected_files:
        if os.path.exists(file):
            abs_path = os.path.abspath(file)
            print(f"  ✅ {file}")
            print(f"      → {abs_path}")
            found_count += 1
        else:
            abs_path = os.path.abspath(file)
            print(f"  ❌ {file} NOT FOUND")
            print(f"      → Expected at: {abs_path}")
    
    if found_count == 3:
        print(f"\n🎉 SUCCESS! All {found_count} images created!")
        print("\n✅ The SHAP display should now work in the app")
        print("\nNext steps:")
        print("1. Restart your Streamlit app")
        print("2. Go to Research Analysis > SHAP XAI Analysis")
        print("3. Select diabetes and click 'Run SHAP Analysis'")
        print("4. Images should now appear!")
    else:
        print(f"\n⚠️  Only {found_count}/3 files found. Check for errors above.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ Test complete!")
