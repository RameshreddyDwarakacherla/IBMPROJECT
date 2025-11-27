"""
Fix the hepatitis model by retraining with correct features
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
import os

print("=" * 70)
print("FIXING HEPATITIS MODEL")
print("=" * 70)

# Change to Frontend directory
os.chdir('Multiple-Disease-Prediction-Webapp/Frontend')

# Load hepatitis data
print("\nLoading hepatitis data...")
df = pd.read_csv('data/hepatitis.csv')
print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"  Columns: {list(df.columns)}")

# Handle missing values
print("\nHandling missing values...")
df = df.replace('?', np.nan)
df = df.apply(pd.to_numeric, errors='coerce')

# Fill missing values with median
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Prepare data
X = df_imputed.iloc[:, :-1].values
y = df_imputed.iloc[:, -1].values.astype(int)

# Ensure labels are 0 and 1
unique_labels = np.unique(y)
if len(unique_labels) == 2 and not (0 in unique_labels and 1 in unique_labels):
    y = np.where(y == unique_labels[0], 0, 1)
    print(f"  Labels remapped: {unique_labels} -> [0, 1]")

print(f"\nData shape:")
print(f"  X: {X.shape}")
print(f"  y: {y.shape}")
print(f"  Unique classes: {np.unique(y)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain/Test split:")
print(f"  Train: {X_train.shape[0]} samples")
print(f"  Test: {X_test.shape[0]} samples")

# Train model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',
    max_depth=10
)

model.fit(X_train, y_train)
print("✓ Model trained successfully")

# Evaluate
print("\nEvaluating model...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")

# Save model
print("\nSaving model...")
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/hepititisc_model.sav')
print("✓ Model saved to: models/hepititisc_model.sav")

# Save metrics
metrics = {
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1)
}

with open('models/hepatitis_model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("✓ Metrics saved to: models/hepatitis_model_metrics.json")

# Test prediction
print("\nTesting prediction...")
test_sample = X_test[0:1]
prediction = model.predict(test_sample)
print(f"✓ Test prediction: {prediction[0]}")

if hasattr(model, 'predict_proba'):
    proba = model.predict_proba(test_sample)
    print(f"✓ Probabilities: {proba[0]}")

print("\n" + "=" * 70)
print("✅ HEPATITIS MODEL FIXED SUCCESSFULLY!")
print("=" * 70)
print(f"\nModel Details:")
print(f"  Features: {X.shape[1]}")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Model Type: RandomForestClassifier")
print(f"  File: models/hepititisc_model.sav")
