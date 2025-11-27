#!/usr/bin/env python3
"""
Check for missing CSV datasets and provide download instructions
"""

import os
import pandas as pd

# Required datasets for the analysis scripts
REQUIRED_DATASETS = {
    'diabetes': {
        'filename': 'diabetes.csv',
        'description': 'Pima Indians Diabetes Dataset',
        'source': 'https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database',
        'alternative': 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv',
        'features': 8,
        'samples': 768
    },
    'heart': {
        'filename': 'heart.csv',
        'description': 'Heart Disease Dataset',
        'source': 'https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset',
        'alternative': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/',
        'features': 13,
        'samples': 303
    },
    'parkinsons': {
        'filename': 'parkinsons.csv',
        'description': 'Parkinsons Disease Dataset',
        'source': 'https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set',
        'alternative': 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/',
        'features': 22,
        'samples': 195
    },
    'liver': {
        'filename': 'indian_liver_patient.csv',
        'description': 'Indian Liver Patient Dataset',
        'source': 'https://www.kaggle.com/datasets/uciml/indian-liver-patient-records',
        'alternative': 'UCI ML Repository',
        'features': 10,
        'samples': 583
    },
    'hepatitis': {
        'filename': 'hepatitis.csv',
        'description': 'Hepatitis Dataset',
        'source': 'https://www.kaggle.com/datasets/codebreaker619/hepatitis-data',
        'alternative': 'https://archive.ics.uci.edu/ml/datasets/hepatitis',
        'features': 19,
        'samples': 155
    },
    'kidney': {
        'filename': 'kidney_disease.csv',
        'description': 'Chronic Kidney Disease Dataset',
        'source': 'https://www.kaggle.com/datasets/mansoordaku/ckdisease',
        'alternative': 'https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease',
        'features': 24,
        'samples': 400
    }
}

DATA_DIR = 'Multiple-Disease-Prediction-Webapp/Frontend/data'

def check_existing_files():
    """Check which files exist and which are missing"""
    
    print("=" * 70)
    print("📊 DATASET STATUS CHECK")
    print("=" * 70)
    
    existing = []
    missing = []
    
    for disease, info in REQUIRED_DATASETS.items():
        filepath = os.path.join(DATA_DIR, info['filename'])
        
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                size_kb = os.path.getsize(filepath) / 1024
                existing.append({
                    'disease': disease,
                    'filename': info['filename'],
                    'rows': df.shape[0],
                    'cols': df.shape[1],
                    'size': size_kb
                })
            except Exception as e:
                print(f"⚠️  {info['filename']} exists but has errors: {e}")
        else:
            missing.append({
                'disease': disease,
                'info': info
            })
    
    # Print existing files
    if existing:
        print(f"\n✅ FOUND ({len(existing)} files):")
        print("-" * 70)
        for item in existing:
            print(f"   • {item['filename']:<30} {item['rows']:>4} rows × {item['cols']:>2} cols  ({item['size']:.1f} KB)")
    
    # Print missing files
    if missing:
        print(f"\n❌ MISSING ({len(missing)} files):")
        print("-" * 70)
        for item in missing:
            disease = item['disease']
            info = item['info']
            print(f"\n   {disease.upper()}: {info['filename']}")
            print(f"      Description: {info['description']}")
            print(f"      Expected: ~{info['samples']} samples, {info['features']} features")
    
    return existing, missing

def generate_download_instructions(missing):
    """Generate instructions for downloading missing datasets"""
    
    if not missing:
        print("\n" + "=" * 70)
        print("🎉 All required datasets are present!")
        print("=" * 70)
        return
    
    print("\n" + "=" * 70)
    print("📥 HOW TO ADD MISSING DATASETS")
    print("=" * 70)
    
    print("\n🔹 METHOD 1: Download from Kaggle (Recommended)")
    print("-" * 70)
    print("1. Go to Kaggle.com and create a free account")
    print("2. Download each dataset:")
    
    for item in missing:
        disease = item['disease']
        info = item['info']
        print(f"\n   {disease.upper()}:")
        print(f"   → {info['source']}")
        print(f"   → Save as: {info['filename']}")
    
    print(f"\n3. Place all CSV files in: {DATA_DIR}")
    
    print("\n\n🔹 METHOD 2: Download from UCI ML Repository")
    print("-" * 70)
    print("Visit: https://archive.ics.uci.edu/ml/datasets.php")
    print("Search for each dataset by name")
    
    print("\n\n🔹 METHOD 3: Use Python to Download (if URLs available)")
    print("-" * 70)
    print("Run the auto-download script:")
    print("   python download_datasets.py")
    
    print("\n\n🔹 METHOD 4: Use Your Own CSV File")
    print("-" * 70)
    print("If you have a CSV file:")
    print("1. Ensure it has the correct format (features + target column)")
    print("2. Run: python add_csv_to_data.py")
    print("3. Follow the prompts to add your file")

def create_download_script(missing):
    """Create a script to auto-download datasets where possible"""
    
    script_content = '''#!/usr/bin/env python3
"""
Auto-download missing datasets from public sources
"""

import urllib.request
import os

DATA_DIR = 'Multiple-Disease-Prediction-Webapp/Frontend/data'

# Direct download URLs (where available)
DOWNLOAD_URLS = {
    'diabetes': 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv',
    'parkinsons': 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data',
}

def download_file(url, filename):
    """Download file from URL"""
    try:
        print(f"Downloading {filename}...")
        filepath = os.path.join(DATA_DIR, filename)
        urllib.request.urlretrieve(url, filepath)
        print(f"✅ Downloaded: {filename}")
        return True
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        return False

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("=" * 60)
    print("Auto-downloading available datasets...")
    print("=" * 60)
    
    # Download diabetes
    if 'diabetes' in DOWNLOAD_URLS:
        download_file(DOWNLOAD_URLS['diabetes'], 'diabetes.csv')
    
    # Download parkinsons
    if 'parkinsons' in DOWNLOAD_URLS:
        download_file(DOWNLOAD_URLS['parkinsons'], 'parkinsons.csv')
    
    print("\\n⚠️  Note: Some datasets require manual download from Kaggle")
    print("   Run: python check_and_download_datasets.py")
    print("   For complete instructions")

if __name__ == "__main__":
    main()
'''
    
    with open('download_datasets.py', 'w') as f:
        f.write(script_content)
    
    print("\n✅ Created: download_datasets.py")

def create_sample_datasets(missing):
    """Create sample/dummy datasets for testing"""
    
    print("\n" + "=" * 70)
    print("🔧 CREATE SAMPLE DATASETS FOR TESTING?")
    print("=" * 70)
    print("\nWould you like to create dummy datasets for testing?")
    print("(These won't be real medical data, just for code testing)")
    print("\nType 'yes' to create sample datasets, or press Enter to skip:")
    
    response = input("   > ").strip().lower()
    
    if response == 'yes':
        import numpy as np
        
        os.makedirs(DATA_DIR, exist_ok=True)
        
        for item in missing:
            disease = item['disease']
            info = item['info']
            
            # Create dummy data
            n_samples = info['samples']
            n_features = info['features']
            
            # Random features
            X = np.random.randn(n_samples, n_features)
            # Binary target
            y = np.random.randint(0, 2, n_samples)
            
            # Create DataFrame
            columns = [f'feature_{i}' for i in range(n_features)] + ['target']
            data = np.column_stack([X, y])
            df = pd.DataFrame(data, columns=columns)
            
            # Save
            filepath = os.path.join(DATA_DIR, info['filename'])
            df.to_csv(filepath, index=False)
            print(f"✅ Created sample: {info['filename']}")
        
        print("\n⚠️  Remember: These are dummy datasets for testing only!")
        print("   Replace with real datasets for actual analysis")

def main():
    existing, missing = check_existing_files()
    
    if missing:
        generate_download_instructions(missing)
        create_download_script(missing)
        create_sample_datasets(missing)
    
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    print(f"✅ Existing datasets: {len(existing)}")
    print(f"❌ Missing datasets: {len(missing)}")
    
    if missing:
        print(f"\n⚠️  You need {len(missing)} more datasets to run the full analysis")
        print("   Follow the instructions above to download them")
    else:
        print("\n🎉 All datasets ready! You can now run:")
        print("   python cross_validation_analysis.py")
        print("   python hyperparameter_tuning_analysis.py")

if __name__ == "__main__":
    main()
