#!/usr/bin/env python3
"""
Helper script to add CSV files to the application data folder
"""

import os
import shutil
import pandas as pd

def add_csv_file(source_path, disease_name=None):
    """
    Copy CSV file to the application data folder
    
    Args:
        source_path: Path to your CSV file
        disease_name: Optional name for the disease (e.g., 'diabetes', 'heart')
    """
    
    # Target directory
    data_dir = 'Multiple-Disease-Prediction-Webapp/Frontend/data'
    
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Get filename
    if disease_name:
        filename = f"{disease_name}.csv"
    else:
        filename = os.path.basename(source_path)
    
    target_path = os.path.join(data_dir, filename)
    
    # Check if source file exists
    if not os.path.exists(source_path):
        print(f"❌ Error: File not found: {source_path}")
        return False
    
    # Validate CSV format
    try:
        df = pd.read_csv(source_path)
        print(f"\n📊 CSV File Info:")
        print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"\n   First few rows:")
        print(df.head(3))
        
        # Check for missing values
        missing = df.isnull().sum().sum()
        if missing > 0:
            print(f"\n⚠️  Warning: {missing} missing values found")
        
        # Check target column (assuming last column)
        target_col = df.iloc[:, -1]
        print(f"\n   Target column distribution:")
        print(target_col.value_counts())
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False
    
    # Copy file
    try:
        shutil.copy2(source_path, target_path)
        print(f"\n✅ Successfully copied to: {target_path}")
        return True
    except Exception as e:
        print(f"❌ Error copying file: {e}")
        return False

def list_current_data_files():
    """List all CSV files currently in the data folder"""
    data_dir = 'Multiple-Disease-Prediction-Webapp/Frontend/data'
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    print(f"\n📁 Current data files in {data_dir}:")
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if files:
        for f in files:
            filepath = os.path.join(data_dir, f)
            size = os.path.getsize(filepath) / 1024  # KB
            print(f"   • {f} ({size:.1f} KB)")
    else:
        print("   (No CSV files found)")

def main():
    print("=" * 60)
    print("CSV File Manager for Disease Prediction App")
    print("=" * 60)
    
    # List current files
    list_current_data_files()
    
    print("\n" + "=" * 60)
    print("How to use this script:")
    print("=" * 60)
    print("\n1. Place your CSV file in the project root directory")
    print("2. Run this script with your filename:")
    print("\n   Example:")
    print("   python add_csv_to_data.py")
    print("\n3. Or use it programmatically:")
    print("\n   from add_csv_to_data import add_csv_file")
    print("   add_csv_file('mydata.csv', 'diabetes')")
    print("\n" + "=" * 60)
    
    # Interactive mode
    print("\n📝 Enter the path to your CSV file (or press Enter to skip):")
    source = input("   > ").strip()
    
    if source:
        print("\n📝 Enter disease name (optional, press Enter to use original filename):")
        disease = input("   > ").strip()
        
        if disease:
            add_csv_file(source, disease)
        else:
            add_csv_file(source)
        
        # Show updated list
        list_current_data_files()

if __name__ == "__main__":
    main()
