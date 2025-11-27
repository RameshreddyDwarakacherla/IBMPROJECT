import os
import shutil

DATA_DIR = 'Multiple-Disease-Prediction-Webapp/Frontend/data'

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print("CSV File Copy Helper")
print("=" * 60)
print("\nThis script will help you copy your CSV files to the data folder.")
print(f"Target folder: {DATA_DIR}")
print("\n" + "=" * 60)

# Required files
required_files = [
    'diabetes.csv',
    'heart.csv', 
    'parkinsons.csv',
    'indian_liver_patient.csv',
    'hepatitis.csv',
    'kidney_disease.csv'
]

print("\nEnter the full path to the folder containing your CSV files:")
print("(Example: C:\\Users\\YourName\\Downloads)")
source_folder = input("\n> ").strip().strip('"')

if not os.path.exists(source_folder):
    print(f"\nError: Folder not found: {source_folder}")
    print("Please check the path and try again.")
    exit(1)

print(f"\nSearching for CSV files in: {source_folder}")
print("-" * 60)

# Find all CSV files in source folder
csv_files = [f for f in os.listdir(source_folder) if f.lower().endswith('.csv')]

if not csv_files:
    print("No CSV files found in that folder!")
    exit(1)

print(f"\nFound {len(csv_files)} CSV files:")
for i, f in enumerate(csv_files, 1):
    print(f"  {i}. {f}")

print("\n" + "=" * 60)
print("Copy Options:")
print("=" * 60)
print("1. Copy ALL CSV files")
print("2. Copy specific files (you'll be asked for each one)")
print("3. Cancel")

choice = input("\nEnter your choice (1-3): ").strip()

copied = 0
skipped = 0

if choice == '1':
    # Copy all files
    print("\nCopying all CSV files...")
    for filename in csv_files:
        source = os.path.join(source_folder, filename)
        dest = os.path.join(DATA_DIR, filename)
        try:
            shutil.copy2(source, dest)
            size_kb = os.path.getsize(dest) / 1024
            print(f"  Copied: {filename} ({size_kb:.1f} KB)")
            copied += 1
        except Exception as e:
            print(f"  Error copying {filename}: {e}")
            skipped += 1

elif choice == '2':
    # Copy specific files
    print("\nFor each file, type 'y' to copy or 'n' to skip:")
    for filename in csv_files:
        response = input(f"\nCopy {filename}? (y/n): ").strip().lower()
        if response == 'y':
            source = os.path.join(source_folder, filename)
            dest = os.path.join(DATA_DIR, filename)
            try:
                shutil.copy2(source, dest)
                size_kb = os.path.getsize(dest) / 1024
                print(f"  Copied: {filename} ({size_kb:.1f} KB)")
                copied += 1
            except Exception as e:
                print(f"  Error: {e}")
                skipped += 1
        else:
            print(f"  Skipped: {filename}")
            skipped += 1
else:
    print("\nCancelled.")
    exit(0)

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Copied: {copied} files")
print(f"Skipped: {skipped} files")

# Check which required files are now present
print("\n" + "=" * 60)
print("Required Files Status:")
print("=" * 60)

for req_file in required_files:
    filepath = os.path.join(DATA_DIR, req_file)
    if os.path.exists(filepath):
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  [OK] {req_file} ({size_kb:.1f} KB)")
    else:
        print(f"  [MISSING] {req_file}")

print("\n" + "=" * 60)
print("Done! Your files are now in:")
print(f"  {os.path.abspath(DATA_DIR)}")
print("=" * 60)
