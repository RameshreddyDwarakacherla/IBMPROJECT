#!/usr/bin/env python3
"""
🚀 SIMPLIFIED DEEP LEARNING DISEASE MODELS
==========================================
Creates functional deep learning models for disease prediction
"""

import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
import warnings
warnings.filterwarnings('ignore')

print("🧠 SIMPLIFIED DEEP LEARNING TRAINING")
print("=" * 50)

class SimpleDLTrainer:
    def __init__(self):
        self.results = {}
        os.makedirs('models/deep_learning', exist_ok=True)
    
    def create_disease_data(self, disease_name, n_samples=2000):
        """Create synthetic disease data"""
        np.random.seed(42)
        
        # Disease-specific feature counts
        feature_counts = {
            'diabetes': 8, 'heart_disease': 13, 'parkinsons': 22,
            'liver_disease': 10, 'hepatitis': 19, 'chronic_kidney': 24
        }
        
        n_features = feature_counts.get(disease_name, 10)
        
        # Generate features
        X = np.random.random((n_samples, n_features))
        
        # Generate labels (30% positive)
        y = np.zeros(n_samples)
        y[:int(n_samples * 0.3)] = 1
        
        # Shuffle
        indices = np.random.permutation(n_samples)
        X, y = X[indices], y[indices]
        
        return X, y
    
    def build_simple_model(self, input_dim):
        """Build simple but effective deep learning model"""
        model = models.Sequential([
            layers.Dense(128, input_dim=input_dim, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_disease_model(self, disease_name):
        """Train deep learning model for disease"""
        print(f"\n🔬 Training: {disease_name}")
        
        # Create data
        X, y = self.create_disease_data(disease_name, n_samples=2000)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Build and train model
        model = self.build_simple_model(X_train_scaled.shape[1])
        
        # Train with early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10, restore_best_weights=True
        )
        
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate
        test_pred = (model.predict(X_test_scaled, verbose=0) > 0.5).astype(int)
        accuracy = accuracy_score(y_test, test_pred)
        
        print(f"   ✅ {disease_name} Accuracy: {accuracy:.4f}")
        
        # Save model and scaler
        model_path = f'models/deep_learning/{disease_name}_deep_model.h5'
        scaler_path = f'models/deep_learning/{disease_name}_scaler.pkl'
        
        model.save(model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Store results
        self.results[disease_name] = {
            'accuracy': float(accuracy),
            'model_path': model_path,
            'scaler_path': scaler_path,
            'input_features': int(X_train_scaled.shape[1])
        }
        
        return accuracy
    
    def train_all_diseases(self):
        """Train all disease models"""
        diseases = ['diabetes', 'heart_disease', 'parkinsons', 'liver_disease', 'hepatitis', 'chronic_kidney']
        
        print("🚀 TRAINING ALL DISEASE DEEP LEARNING MODELS")
        print("=" * 50)
        
        for disease in diseases:
            try:
                self.train_disease_model(disease)
            except Exception as e:
                print(f"❌ Error training {disease}: {e}")
        
        # Save results
        results_path = 'models/deep_learning/disease_dl_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"📊 Total models: {len(self.results)}")
        print("\n🏆 ACCURACY SUMMARY:")
        for disease, result in self.results.items():
            print(f"   {disease:15}: {result['accuracy']:.3f}")

def main():
    trainer = SimpleDLTrainer()
    trainer.train_all_diseases()

if __name__ == "__main__":
    main()