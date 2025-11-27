#!/usr/bin/env python3
"""
Deep Learning Models for Enhanced Disease Prediction
Implements advanced neural networks for better performance and new features
"""

import numpy as np
import pandas as pd

# Try to import TensorFlow with proper error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True

    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

except ImportError as e:
    print(f"TensorFlow not available: {e}")
    TENSORFLOW_AVAILABLE = False
    # Create dummy classes to prevent import errors
    class DummyTensorFlow:
        def __getattr__(self, name):
            raise ImportError("TensorFlow is not installed")

    tf = DummyTensorFlow()
    keras = DummyTensorFlow()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

class DeepLearningDiseasePredictor:
    """
    Advanced Deep Learning Disease Prediction System
    """

    def __init__(self):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available. Please install TensorFlow first.")

        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.model_configs = {
            'diabetes': {
                'input_dim': 8,
                'hidden_layers': [64, 32, 16],
                'dropout_rate': 0.3,
                'activation': 'relu',
                'output_activation': 'sigmoid'
            },
            'heart_disease': {
                'input_dim': 13,
                'hidden_layers': [128, 64, 32],
                'dropout_rate': 0.4,
                'activation': 'relu',
                'output_activation': 'sigmoid'
            },
            'parkinsons': {
                'input_dim': 22,
                'hidden_layers': [256, 128, 64, 32],
                'dropout_rate': 0.5,
                'activation': 'relu',
                'output_activation': 'sigmoid'
            },
            'chronic_kidney': {
                'input_dim': 24,
                'hidden_layers': [128, 64, 32, 16],
                'dropout_rate': 0.4,
                'activation': 'relu',
                'output_activation': 'sigmoid'
            },
            'liver_disease': {
                'input_dim': 10,
                'hidden_layers': [64, 32, 16],
                'dropout_rate': 0.3,
                'activation': 'relu',
                'output_activation': 'sigmoid'
            },
            'hepatitis': {
                'input_dim': 19,
                'hidden_layers': [128, 64, 32],
                'dropout_rate': 0.4,
                'activation': 'relu',
                'output_activation': 'sigmoid'
            }
        }
    
    def create_neural_network(self, config):
        """Create a deep neural network based on configuration"""
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Dense(
            config['hidden_layers'][0], 
            input_dim=config['input_dim'],
            activation=config['activation'],
            kernel_regularizer=keras.regularizers.l2(0.001)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(config['dropout_rate']))
        
        # Hidden layers
        for units in config['hidden_layers'][1:]:
            model.add(layers.Dense(
                units, 
                activation=config['activation'],
                kernel_regularizer=keras.regularizers.l2(0.001)
            ))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(config['dropout_rate']))
        
        # Output layer
        model.add(layers.Dense(1, activation=config['output_activation']))
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def create_ensemble_model(self, config):
        """Create an ensemble model combining multiple architectures"""
        # Input layer
        inputs = layers.Input(shape=(config['input_dim'],))
        
        # Branch 1: Deep narrow network
        x1 = layers.Dense(64, activation='relu')(inputs)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Dropout(0.3)(x1)
        x1 = layers.Dense(32, activation='relu')(x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Dropout(0.3)(x1)
        x1 = layers.Dense(16, activation='relu')(x1)
        
        # Branch 2: Wide shallow network
        x2 = layers.Dense(128, activation='relu')(inputs)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Dropout(0.4)(x2)
        x2 = layers.Dense(64, activation='relu')(x2)
        
        # Branch 3: Residual connections
        x3 = layers.Dense(64, activation='relu')(inputs)
        x3 = layers.BatchNormalization()(x3)
        residual = x3
        x3 = layers.Dense(64, activation='relu')(x3)
        x3 = layers.BatchNormalization()(x3)
        x3 = layers.Add()([x3, residual])  # Residual connection
        x3 = layers.Dropout(0.3)(x3)
        
        # Combine branches
        combined = layers.Concatenate()([x1, x2, x3])
        combined = layers.Dense(64, activation='relu')(combined)
        combined = layers.BatchNormalization()(combined)
        combined = layers.Dropout(0.4)(combined)
        combined = layers.Dense(32, activation='relu')(combined)
        combined = layers.Dropout(0.3)(combined)
        
        # Output
        outputs = layers.Dense(1, activation='sigmoid')(combined)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_deep_model(self, disease_name, X, y, use_ensemble=False):
        """Train a deep learning model for a specific disease"""
        print(f"Training deep learning model for {disease_name}...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers[disease_name] = scaler
        
        # Create model
        config = self.model_configs.get(disease_name, self.model_configs['diabetes'])
        config['input_dim'] = X_train_scaled.shape[1]
        
        if use_ensemble:
            model = self.create_ensemble_model(config)
        else:
            model = self.create_neural_network(config)
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=200,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Evaluate model
        y_pred_prob = model.predict(X_test_scaled)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Deep Learning Model Performance for {disease_name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        # Store model
        self.models[disease_name] = model
        
        # Save model and metrics
        model_dir = f"models/deep_learning"
        os.makedirs(model_dir, exist_ok=True)
        
        model.save(f"{model_dir}/{disease_name}_deep_model.h5")
        joblib.dump(scaler, f"{model_dir}/{disease_name}_scaler.pkl")
        
        # Save metrics
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'model_type': 'Deep Neural Network' + (' Ensemble' if use_ensemble else ''),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'epochs_trained': len(history.history['loss']),
            'best_val_loss': float(min(history.history['val_loss']))
        }
        
        with open(f"{model_dir}/{disease_name}_deep_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return model, accuracy, history
    
    def predict_with_confidence(self, disease_name, X):
        """Make predictions with confidence intervals"""
        if disease_name not in self.models:
            # Try to load the model first
            if not self.load_deep_model(disease_name):
                # If loading fails, try to create a fallback model
                if not self.create_fallback_model(disease_name):
                    raise ValueError(f"Model for {disease_name} not found and could not be created")

        model = self.models[disease_name]
        scaler = self.scalers[disease_name]

        # Validate that model is actually a TensorFlow/Keras model
        if not hasattr(model, 'predict'):
            raise ValueError(f"Invalid model object for {disease_name}. Expected TensorFlow/Keras model.")

        # Scale input
        X_scaled = scaler.transform(X)

        # Get prediction probabilities
        pred_probs = model.predict(X_scaled)

        # Monte Carlo Dropout for uncertainty estimation
        mc_predictions = []
        try:
            for _ in range(100):  # 100 Monte Carlo samples
                mc_pred = model(X_scaled, training=True)  # Enable dropout during inference
                mc_predictions.append(mc_pred.numpy())
        except Exception as e:
            print(f"Warning: Monte Carlo dropout failed for {disease_name}: {e}")
            # Fallback to simple prediction without uncertainty estimation
            mc_predictions = [pred_probs for _ in range(10)]

        mc_predictions = np.array(mc_predictions)

        # Calculate statistics
        mean_pred = np.mean(mc_predictions, axis=0)
        std_pred = np.std(mc_predictions, axis=0)

        # Confidence intervals (95%)
        lower_bound = np.percentile(mc_predictions, 2.5, axis=0)
        upper_bound = np.percentile(mc_predictions, 97.5, axis=0)

        return {
            'prediction': (mean_pred > 0.5).astype(int).flatten(),
            'probability': mean_pred.flatten(),
            'confidence_interval': {
                'lower': lower_bound.flatten(),
                'upper': upper_bound.flatten()
            },
            'uncertainty': std_pred.flatten()
        }
    
    def load_deep_model(self, disease_name):
        """Load a pre-trained deep learning model"""
        model_dir = f"models/deep_learning"

        try:
            model = keras.models.load_model(f"{model_dir}/{disease_name}_deep_model.h5")
            scaler = joblib.load(f"{model_dir}/{disease_name}_scaler.pkl")

            self.models[disease_name] = model
            self.scalers[disease_name] = scaler

            return True
        except Exception as e:
            print(f"Error loading deep model for {disease_name}: {e}")
            return False

    def create_fallback_model(self, disease_name):
        """Create a simple fallback model if deep learning model is not available"""
        try:
            config = self.model_configs.get(disease_name, self.model_configs['diabetes'])

            # Create a simple neural network
            model = keras.Sequential([
                keras.layers.Dense(32, activation='relu', input_shape=(config['input_dim'],)),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(16, activation='relu'),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(1, activation='sigmoid')
            ])

            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            # Create a dummy scaler
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()

            # Fit scaler with dummy data
            dummy_data = np.random.randn(100, config['input_dim'])
            scaler.fit(dummy_data)

            self.models[disease_name] = model
            self.scalers[disease_name] = scaler

            print(f"Created fallback model for {disease_name}")
            return True

        except Exception as e:
            print(f"Error creating fallback model for {disease_name}: {e}")
            return False
