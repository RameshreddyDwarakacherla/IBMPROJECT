import os
import shutil

DATA_DIR = 'Multiple-Disease-Prediction-Webapp/Frontend/data'

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Mapping of your file names to required names
FILE_MAPPING = {
    'diabetes.csv': 'diabetes.csv',
    'heart_Disease_Prediction.csv': 'heart.csv',  # Will be renamed
    'parkinsons.csv': 'parkinsons.csv',
    'indian_liver_patient.csv': 'indian_liver_patient.csv',
    'hepatitis.csv': 'hepatitis.csv',
    'kidney_disease.csv': 'kidney_disease.csv'
}

print("=" * 70)
print("Adding Your CSV Files to Data Folder")
print("=" * 70)

print("\nWhere are your CSV files located?")
print("Enter the full path to the folder:")
print("(Example: C:\\Users\\YourName\\Downloads)")
print("Or press Enter if they're in the current folder")

source_folder = input("\n> ").strip().strip('"')

if not source_folder:
    source_folder = '.'

if not os.path.exists(source_folder):
    print(f"\nError: Folder not found: {source_folder}")
    exit(1)

print(f"\nLooking for files in: {os.path.abspath(source_folder)}")
print("-" * 70)

copied = 0
missing = []

for source_name, target_name in FILE_MAPPING.items():
    source_path = os.path.join(source_folder, source_name)
    target_path = os.path.join(DATA_DIR, target_name)
    
    if os.path.exists(source_path):
        try:
            shutil.copy2(source_path, target_path)
            size_kb = os.path.getsize(target_path) / 1024
            
            if source_name != target_name:
                print(f"  [OK] {source_name} -> {target_name} ({size_kb:.1f} KB)")
            else:
                print(f"  [OK] {source_name} ({size_kb:.1f} KB)")
            
            copied += 1
        except Exception as e:
            print(f"  [ERROR] {source_name}: {e}")
    else:
        print(f"  [NOT FOUND] {source_name}")
        missing.append(source_name)

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"Successfully copied: {copied} files")
print(f"Missing: {len(missing)} files")

if missing:
    print("\nMissing files:")
    for f in missing:
        print(f"  - {f}")

print("\n" + "=" * 70)
print("Files are now in:")
print(f"  {os.path.abspath(DATA_DIR)}")
print("=" * 70)

if copied == len(FILE_MAPPING):
    print("\n SUCCESS! All datasets are ready!")
    print("\nYou can now run:")
    print("  python cross_validation_analysis.py")
    print("  python hyperparameter_tuning_analysis.py")
else:
    print(f"\n WARNING: Only {copied} of {len(FILE_MAPPING)} files were copied.")
    print("Please check the file locations and try again.")
