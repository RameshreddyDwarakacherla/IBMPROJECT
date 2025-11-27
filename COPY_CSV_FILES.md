# How to Add Your Downloaded CSV Files

## Quick Method (Copy-Paste in File Explorer)

1. Open File Explorer
2. Navigate to where you downloaded the CSV files
3. Select all the CSV files you want to add
4. Copy them (Ctrl+C)
5. Navigate to: `Multiple-Disease-Prediction-Webapp\Frontend\data`
6. Paste them (Ctrl+V)

## Required File Names

Make sure your files are named exactly as follows:

- diabetes.csv
- heart_Disease_Prediction.csv
- parkinsons.csv
- indian_liver_patient.csv
- hepatitis.csv
- kidney_disease.csv

If your files have different names, rename them to match the above.

## Using Command Line (Windows)

Open Command Prompt in your project folder and run:

```cmd
copy "path\to\your\diabetes.csv" "Multiple-Disease-Prediction-Webapp\Frontend\data\diabetes.csv"
copy "path\to\your\heart.csv" "Multiple-Disease-Prediction-Webapp\Frontend\data\heart.csv"
copy "path\to\your\parkinsons.csv" "Multiple-Disease-Prediction-Webapp\Frontend\data\parkinsons.csv"
```

Replace "path\to\your\" with the actual location of your files.
