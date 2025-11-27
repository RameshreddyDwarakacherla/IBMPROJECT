#!/usr/bin/env python3
"""
🚀 COMPLETE DEEP LEARNING MODEL TRAINER
===================================
Trains comprehensive deep learning models for all diseases using TensorFlow/Keras
with state-of-the-art architectures and optimization techniques.
"""

import os
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

print("🧠 DEEP LEARNING MODEL TRAINING SYSTEM")
print("=" * 60)
print(f"TensorFlow Version: {tf.__version__}")
try:
    print(f"Keras Version: {tf.keras.__version__}")
except:
    print("Keras Version: Integrated with TensorFlow")
print("=" * 60)

class DeepLearningTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directories"""
        os.makedirs('models/deep_learning', exist_ok=True)
        print("✅ Created deep learning models directory")
    
    def create_synthetic_data(self, disease_name, n_samples=2000):
        """Create synthetic medical data for training"""
        np.random.seed(42)
        
        # Define feature configurations for different diseases
        feature_configs = {
            'diabetes': {
                'n_features': 8,
                'feature_names': ['glucose', 'blood_pressure', 'insulin', 'bmi', 'age', 'pregnancies', 'skin_thickness', 'pedigree'],
                'ranges': [(80, 200), (60, 140), (15, 200), (18, 45), (21, 70), (0, 10), (10, 50), (0.1, 2.0)]
            },
            'heart_disease': {
                'n_features': 13,
                'feature_names': ['age', 'sex', 'chest_pain', 'blood_pressure', 'cholesterol', 'fasting_sugar', 'ecg', 'max_heart_rate', 'angina', 'depression', 'slope', 'vessels', 'defect'],
                'ranges': [(29, 77), (0, 1), (0, 3), (94, 200), (126, 564), (0, 1), (0, 2), (71, 202), (0, 1), (0, 6.2), (0, 2), (0, 4), (0, 3)]
            },
            'parkinsons': {
                'n_features': 22,
                'feature_names': ['fo', 'fhi', 'flo', 'jitter_percent', 'jitter_abs', 'rap', 'ppq', 'ddp', 'shimmer', 'shimmer_db', 'apq3', 'apq5', 'apq', 'dda', 'nhr', 'hnr', 'rpde', 'dfa', 'spread1', 'spread2', 'd2', 'ppe'],
                'ranges': [(88, 260), (102, 592), (65, 239), (0.00168, 0.03316), (0.000007, 0.000260), (0.00068, 0.02144), (0.00092, 0.01394), (0.00204, 0.06433), (0.00954, 0.11908), (0.085, 1.302), (0.00455, 0.05648), (0.00532, 0.07950), (0.01024, 0.13926), (0.01364, 0.16944), (0.00065, 0.75197), (8.441, 33.047), (0.151, 0.685), (0.574, 0.825), (-7.964, -2.434), (0.006, 0.450), (1.423, 12.863), (0.044, 0.527)]
            },
            'liver_disease': {
                'n_features': 10,
                'feature_names': ['age', 'gender', 'total_bilirubin', 'direct_bilirubin', 'alkaline_phosphotase', 'alamine_aminotransferase', 'aspartate_aminotransferase', 'total_proteins', 'albumin', 'albumin_globulin_ratio'],
                'ranges': [(10, 90), (0, 1), (0.4, 75), (0.1, 19.7), (63, 2110), (10, 2000), (10, 4929), (2.7, 9.6), (0.9, 5.5), (0.3, 2.8)]
            },
            'hepatitis': {
                'n_features': 19,
                'feature_names': ['age', 'sex', 'steroid', 'antivirals', 'fatigue', 'malaise', 'anorexia', 'liver_big', 'liver_firm', 'spleen_palpable', 'spiders', 'ascites', 'varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime', 'histology'],
                'ranges': [(7, 78), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0.3, 8.0), (26, 295), (14, 648), (2.1, 6.4), (0, 100), (0, 1)]
            },
            'chronic_kidney': {
                'n_features': 24,
                'feature_names': ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium', 'potassium', 'hemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count', 'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'pedal_edema', 'anemia'],
                'ranges': [(2, 90), (50, 180), (1.005, 1.025), (0, 5), (0, 5), (0, 1), (0, 1), (0, 1), (0, 1), (22, 490), (1.5, 391), (0.4, 76), (4.5, 163), (2.5, 47), (3.1, 17.8), (9, 54), (2200, 26400), (2.1, 8.0), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
            }
        }
        
        config = feature_configs.get(disease_name, feature_configs['diabetes'])
        n_features = config['n_features']
        feature_names = config['feature_names']
        ranges = config['ranges']
        
        # Generate features
        X = np.zeros((n_samples, n_features))
        for i, (min_val, max_val) in enumerate(ranges):
            if i < len(ranges):
                X[:, i] = np.random.uniform(min_val, max_val, n_samples)
        
        # Generate labels with realistic distribution
        positive_ratio = 0.3  # 30% positive cases
        n_positive = int(n_samples * positive_ratio)
        y = np.zeros(n_samples)
        y[:n_positive] = 1
        
        # Add some correlation between features and labels
        for i in range(n_positive):
            # Make positive cases have more extreme values
            X[i] *= np.random.uniform(1.1, 1.5)
        
        # Shuffle the data
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y.astype(int)
        
        print(f"✅ Created {n_samples} samples for {disease_name}")
        print(f"   Features: {n_features}, Positive cases: {n_positive} ({positive_ratio*100:.1f}%)")
        
        return df
    
    def build_deep_model(self, input_dim, model_type='classification'):
        """Build optimized deep neural network"""
        model = models.Sequential([
            # Input layer with batch normalization
            layers.Dense(256, input_dim=input_dim),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            
            # Hidden layers with residual connections simulation
            layers.Dense(128),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.25),
            
            layers.Dense(64),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.2),
            
            layers.Dense(32),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.15),
            
            # Output layer
            layers.Dense(1 if model_type == 'classification' else 2, 
                        activation='sigmoid' if model_type == 'classification' else 'softmax')
        ])
        
        # Compile with advanced optimizer
        optimizer = optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy' if model_type == 'classification' else 'sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def build_ensemble_model(self, input_dim):
        """Build ensemble deep learning model"""
        # Create multiple sub-models
        inputs = layers.Input(shape=(input_dim,))
        
        # Branch 1: Deep network
        branch1 = layers.Dense(128, activation='relu')(inputs)
        branch1 = layers.BatchNormalization()(branch1)
        branch1 = layers.Dropout(0.3)(branch1)
        branch1 = layers.Dense(64, activation='relu')(branch1)
        branch1 = layers.BatchNormalization()(branch1)
        branch1 = layers.Dropout(0.2)(branch1)
        
        # Branch 2: Wide network
        branch2 = layers.Dense(256, activation='relu')(inputs)
        branch2 = layers.BatchNormalization()(branch2)
        branch2 = layers.Dropout(0.25)(branch2)
        branch2 = layers.Dense(32, activation='relu')(branch2)
        
        # Combine branches
        combined = layers.concatenate([branch1, branch2])
        combined = layers.Dense(32, activation='relu')(combined)
        combined = layers.Dropout(0.15)(combined)
        outputs = layers.Dense(1, activation='sigmoid')(combined)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_disease_model(self, disease_name):
        """Train deep learning model for specific disease"""
        print(f"\n🔬 Training Deep Learning Model: {disease_name.upper()}")
        print("-" * 50)
        
        # Create synthetic data
        df = self.create_synthetic_data(disease_name, n_samples=3000)
        
        # Prepare data
        X = df.drop('target', axis=1).values
        y = df['target'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Calculate class weights for balanced training
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        # Build models
        print("🏗️  Building Deep Neural Network...")
        deep_model = self.build_deep_model(X_train_scaled.shape[1])
        
        print("🏗️  Building Ensemble Model...")
        ensemble_model = self.build_ensemble_model(X_train_scaled.shape[1])
        
        # Define callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train deep model
        print("🚀 Training Deep Neural Network...")
        deep_history = deep_model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=100,
            batch_size=32,
            class_weight=class_weight_dict,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Train ensemble model
        print("🚀 Training Ensemble Model...")
        ensemble_history = ensemble_model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=80,
            batch_size=32,
            class_weight=class_weight_dict,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate models
        deep_pred = (deep_model.predict(X_test_scaled) > 0.5).astype(int)
        ensemble_pred = (ensemble_model.predict(X_test_scaled) > 0.5).astype(int)
        
        deep_accuracy = accuracy_score(y_test, deep_pred)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        
        print(f"\n📊 RESULTS for {disease_name}:")
        print(f"   Deep Model Accuracy: {deep_accuracy:.4f}")
        print(f"   Ensemble Model Accuracy: {ensemble_accuracy:.4f}")
        
        # Save models
        deep_model_path = f'models/deep_learning/{disease_name}_deep_model.h5'
        ensemble_model_path = f'models/deep_learning/{disease_name}_ensemble_model.h5'
        scaler_path = f'models/deep_learning/{disease_name}_scaler.pkl'
        
        deep_model.save(deep_model_path)
        ensemble_model.save(ensemble_model_path)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Store results
        self.results[disease_name] = {
            'deep_accuracy': float(deep_accuracy),
            'ensemble_accuracy': float(ensemble_accuracy),
            'deep_model_path': deep_model_path,
            'ensemble_model_path': ensemble_model_path,
            'scaler_path': scaler_path,
            'input_features': int(X_train_scaled.shape[1]),
            'training_samples': int(len(X_train)),
            'test_samples': int(len(X_test))
        }
        
        print(f"✅ Models saved successfully!")
        return deep_accuracy, ensemble_accuracy
    
    def train_medical_image_models(self):
        """Train medical image analysis models"""
        print(f"\n🖼️  Training Medical Image Analysis Models")
        print("-" * 50)
        
        image_types = ['chest_xray', 'brain_mri', 'retinal_scan', 'skin_lesion']
        
        for image_type in image_types:
            print(f"\n🔬 Training {image_type} model...")
            
            # Create synthetic medical image features (simulating CNN features)
            n_samples = 2000
            if image_type == 'chest_xray':
                n_features = 2048  # ResNet features
            elif image_type == 'brain_mri':
                n_features = 1024  # VGG features  
            elif image_type == 'retinal_scan':
                n_features = 512   # MobileNet features
            else:  # skin_lesion
                n_features = 256   # EfficientNet features
            
            # Generate synthetic CNN features
            np.random.seed(42)
            X = np.random.random((n_samples, n_features))
            
            # Create realistic labels (30% positive)
            y = np.zeros(n_samples)
            y[:int(n_samples * 0.3)] = 1
            
            # Shuffle
            indices = np.random.permutation(n_samples)
            X, y = X[indices], y[indices]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Build image classification model
            model = models.Sequential([
                layers.Dense(512, input_dim=n_features, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.1),
                
                layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            history = model.fit(
                X_train_scaled, y_train,
                validation_data=(X_test_scaled, y_test),
                epochs=50,
                batch_size=32,
                callbacks=[
                    callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
                ],
                verbose=1
            )
            
            # Evaluate
            test_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
            accuracy = accuracy_score(y_test, test_pred)
            
            print(f"   {image_type} Model Accuracy: {accuracy:.4f}")
            
            # Save model
            model_path = f'models/deep_learning/{image_type}_image_model.h5'
            scaler_path = f'models/deep_learning/{image_type}_scaler.pkl'
            
            model.save(model_path)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            # Store results
            self.results[f'{image_type}_image'] = {
                'accuracy': float(accuracy),
                'model_path': model_path,
                'scaler_path': scaler_path,
                'input_features': int(n_features),
                'model_type': 'image_classification'
            }
    
    def save_training_results(self):
        """Save comprehensive training results"""
        results_path = 'models/deep_learning/training_results.json'
        
        # Add metadata
        self.results['training_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'tensorflow_version': tf.__version__,
            'keras_version': 'Integrated with TensorFlow',
            'total_models_trained': len([k for k in self.results.keys() if k != 'training_metadata'])
        }
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Training results saved to: {results_path}")
    
    def train_all_models(self):
        """Train all deep learning models"""
        print("🚀 STARTING COMPREHENSIVE DEEP LEARNING TRAINING")
        print("=" * 60)
        
        diseases = ['diabetes', 'heart_disease', 'parkinsons', 'liver_disease', 'hepatitis', 'chronic_kidney']
        
        # Train disease prediction models
        for disease in diseases:
            try:
                self.train_disease_model(disease)
            except Exception as e:
                print(f"❌ Error training {disease}: {e}")
                continue
        
        # Train medical image models
        try:
            self.train_medical_image_models()
        except Exception as e:
            print(f"❌ Error training image models: {e}")
        
        # Save results
        self.save_training_results()
        
        print("\n🎉 DEEP LEARNING TRAINING COMPLETED!")
        print("=" * 60)
        print(f"📊 Total models trained: {len([k for k in self.results.keys() if k != 'training_metadata'])}")
        print("\n🏆 PERFORMANCE SUMMARY:")
        for disease in diseases:
            if disease in self.results:
                deep_acc = self.results[disease]['deep_accuracy']
                ensemble_acc = self.results[disease]['ensemble_accuracy']
                print(f"   {disease:15}: Deep={deep_acc:.3f}, Ensemble={ensemble_acc:.3f}")

def main():
    """Main training function"""
    trainer = DeepLearningTrainer()
    trainer.train_all_models()

if __name__ == "__main__":
    main()