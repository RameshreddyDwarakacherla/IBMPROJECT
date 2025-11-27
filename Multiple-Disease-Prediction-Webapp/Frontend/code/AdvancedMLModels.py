#!/usr/bin/env python3
"""
Advanced Machine Learning Models (TensorFlow-Free Version)
Enhanced ML models using scikit-learn, XGBoost, and LightGBM for better performance
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

class AdvancedMLPredictor:
    """
    Advanced Machine Learning Disease Prediction System
    Uses ensemble methods and hyperparameter optimization for better performance
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.best_params = {}
        self.model_configs = {
            'diabetes': {
                'features': 8,
                'algorithms': ['rf', 'xgb', 'lgb', 'svm'],
                'target_accuracy': 0.85
            },
            'heart_disease': {
                'features': 13,
                'algorithms': ['rf', 'xgb', 'lgb', 'lr'],
                'target_accuracy': 0.88
            },
            'parkinsons': {
                'features': 22,
                'algorithms': ['rf', 'xgb', 'svm', 'gb'],
                'target_accuracy': 0.90
            },
            'chronic_kidney': {
                'features': 24,
                'algorithms': ['rf', 'xgb', 'lgb', 'lr'],
                'target_accuracy': 0.85
            },
            'liver_disease': {
                'features': 10,
                'algorithms': ['rf', 'xgb', 'lgb', 'svm'],
                'target_accuracy': 0.82
            },
            'hepatitis': {
                'features': 19,
                'algorithms': ['rf', 'xgb', 'gb', 'lr'],
                'target_accuracy': 0.88
            }
        }
    
    def create_advanced_models(self, disease_name):
        """Create advanced ML models with hyperparameter optimization"""
        models = {}
        
        # Random Forest with optimized parameters
        models['rf'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        
        # XGBoost with optimized parameters
        models['xgb'] = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        # LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            models['lgb'] = lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
        
        # Gradient Boosting
        models['gb'] = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        # Support Vector Machine
        models['svm'] = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
        
        # Logistic Regression
        models['lr'] = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        return models
    
    def create_ensemble_model(self, base_models):
        """Create ensemble model combining multiple algorithms"""
        # Select best performing models for ensemble
        ensemble_models = []
        
        for name, model in base_models.items():
            ensemble_models.append((name, model))
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'  # Use probability-based voting
        )
        
        return ensemble
    
    def hyperparameter_optimization(self, model, X, y, param_grid):
        """Perform hyperparameter optimization using GridSearchCV"""
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        return grid_search.best_estimator_, grid_search.best_params_
    
    def train_advanced_model(self, disease_name, X, y, optimize_hyperparams=True):
        """Train advanced ML model for a specific disease"""
        print(f"Training advanced ML model for {disease_name}...")
        
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
        
        # Create base models
        base_models = self.create_advanced_models(disease_name)
        config = self.model_configs.get(disease_name, self.model_configs['diabetes'])
        
        # Filter models based on configuration
        selected_models = {k: v for k, v in base_models.items() if k in config['algorithms']}
        
        # Train and evaluate individual models
        model_scores = {}
        trained_models = {}
        
        for name, model in selected_models.items():
            try:
                # Hyperparameter optimization (optional)
                if optimize_hyperparams and name in ['rf', 'xgb']:
                    if name == 'rf':
                        param_grid = {
                            'n_estimators': [100, 150, 200],
                            'max_depth': [10, 15, 20],
                            'min_samples_split': [2, 5, 10]
                        }
                    elif name == 'xgb':
                        param_grid = {
                            'n_estimators': [100, 150, 200],
                            'max_depth': [6, 8, 10],
                            'learning_rate': [0.05, 0.1, 0.15]
                        }
                    
                    optimized_model, best_params = self.hyperparameter_optimization(
                        model, X_train_scaled, y_train, param_grid
                    )
                    self.best_params[f"{disease_name}_{name}"] = best_params
                    model = optimized_model
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                model_scores[name] = {'accuracy': accuracy, 'f1_score': f1}
                trained_models[name] = model
                
                print(f"  {name.upper()}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
        
        # Create ensemble model
        if len(trained_models) >= 2:
            ensemble_model = self.create_ensemble_model(trained_models)
            ensemble_model.fit(X_train_scaled, y_train)
            
            # Evaluate ensemble
            y_pred_ensemble = ensemble_model.predict(X_test_scaled)
            ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
            ensemble_f1 = f1_score(y_test, y_pred_ensemble, average='weighted')
            
            print(f"  ENSEMBLE: Accuracy = {ensemble_accuracy:.4f}, F1 = {ensemble_f1:.4f}")
            
            # Use ensemble if it performs better than individual models
            best_individual_score = max([scores['f1_score'] for scores in model_scores.values()])
            if ensemble_f1 > best_individual_score:
                final_model = ensemble_model
                final_accuracy = ensemble_accuracy
                final_f1 = ensemble_f1
                model_type = 'Ensemble'
            else:
                # Use best individual model
                best_model_name = max(model_scores.items(), key=lambda x: x[1]['f1_score'])[0]
                final_model = trained_models[best_model_name]
                final_accuracy = model_scores[best_model_name]['accuracy']
                final_f1 = model_scores[best_model_name]['f1_score']
                model_type = best_model_name.upper()
        else:
            # Use single best model
            best_model_name = max(model_scores.items(), key=lambda x: x[1]['f1_score'])[0]
            final_model = trained_models[best_model_name]
            final_accuracy = model_scores[best_model_name]['accuracy']
            final_f1 = model_scores[best_model_name]['f1_score']
            model_type = best_model_name.upper()
        
        # Store final model
        self.models[disease_name] = final_model
        
        # Calculate additional metrics
        y_pred_final = final_model.predict(X_test_scaled)
        precision = precision_score(y_test, y_pred_final, average='weighted')
        recall = recall_score(y_test, y_pred_final, average='weighted')
        
        print(f"Final Model ({model_type}) Performance:")
        print(f"  Accuracy: {final_accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {final_f1:.4f}")
        
        # Save model and metrics
        model_dir = f"models/advanced_ml"
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(final_model, f"{model_dir}/{disease_name}_advanced_model.pkl")
        joblib.dump(scaler, f"{model_dir}/{disease_name}_scaler.pkl")
        
        # Save metrics
        metrics = {
            'accuracy': float(final_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(final_f1),
            'model_type': model_type,
            'individual_scores': model_scores,
            'best_params': self.best_params.get(f"{disease_name}_{model_type.lower()}", {}),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        with open(f"{model_dir}/{disease_name}_advanced_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return final_model, final_accuracy
    
    def predict_with_confidence(self, disease_name, X):
        """Make predictions with confidence scores"""
        if disease_name not in self.models:
            # Try to load the model if it's not already loaded
            if not self.load_advanced_model(disease_name):
                raise ValueError(f"Model for {disease_name} not found and could not be loaded")

        model = self.models[disease_name]
        scaler = self.scalers[disease_name]
        
        # Scale input
        X_scaled = scaler.transform(X)
        
        # Get predictions
        predictions = model.predict(X_scaled)
        
        # Get prediction probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_scaled)
            confidence = np.max(probabilities, axis=1)
            class_probabilities = probabilities
        else:
            # For models without predict_proba, use decision function
            decision_scores = model.decision_function(X_scaled)
            confidence = np.abs(decision_scores)
            class_probabilities = None
        
        return {
            'predictions': predictions,
            'confidence': confidence,
            'probabilities': class_probabilities
        }
    
    def load_advanced_model(self, disease_name):
        """Load a pre-trained advanced ML model"""
        model_dir = f"models/advanced_ml"
        
        try:
            model = joblib.load(f"{model_dir}/{disease_name}_advanced_model.pkl")
            scaler = joblib.load(f"{model_dir}/{disease_name}_scaler.pkl")
            
            self.models[disease_name] = model
            self.scalers[disease_name] = scaler
            
            return True
        except Exception as e:
            print(f"Error loading advanced model for {disease_name}: {e}")
            return False
