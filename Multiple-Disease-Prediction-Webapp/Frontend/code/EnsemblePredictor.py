#!/usr/bin/env python3
"""
Ensemble Disease Prediction System
Combines traditional ML models with deep learning for superior performance
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
import os
from .DeepLearningModels import DeepLearningDiseasePredictor
from .MedicalImageAnalysis import MedicalImageAnalyzer
import warnings
warnings.filterwarnings('ignore')

class EnsembleDiseasePredictor:
    """
    Advanced Ensemble System combining multiple prediction approaches
    """
    
    def __init__(self):
        self.traditional_models = {}
        self.deep_models = {}
        self.ensemble_models = {}
        self.image_analyzer = MedicalImageAnalyzer()
        self.deep_predictor = DeepLearningDiseasePredictor()
        self.scalers = {}
        
        # Ensemble configurations
        self.ensemble_configs = {
            'diabetes': {
                'traditional_weight': 0.3,
                'deep_weight': 0.5,
                'ensemble_weight': 0.2,
                'confidence_threshold': 0.7
            },
            'heart_disease': {
                'traditional_weight': 0.4,
                'deep_weight': 0.4,
                'ensemble_weight': 0.2,
                'confidence_threshold': 0.75
            },
            'parkinsons': {
                'traditional_weight': 0.2,
                'deep_weight': 0.6,
                'ensemble_weight': 0.2,
                'confidence_threshold': 0.65
            },
            'chronic_kidney': {
                'traditional_weight': 0.3,
                'deep_weight': 0.5,
                'ensemble_weight': 0.2,
                'confidence_threshold': 0.7
            },
            'liver_disease': {
                'traditional_weight': 0.4,
                'deep_weight': 0.4,
                'ensemble_weight': 0.2,
                'confidence_threshold': 0.8
            },
            'hepatitis': {
                'traditional_weight': 0.3,
                'deep_weight': 0.5,
                'ensemble_weight': 0.2,
                'confidence_threshold': 0.75
            }
        }
    
    def load_traditional_models(self):
        """Load existing traditional ML models"""
        model_files = {
            'diabetes': 'models/diabetes_model.sav',
            'heart_disease': 'models/heart_disease_model.sav',
            'parkinsons': 'models/parkinsons_model.sav',
            'chronic_kidney': 'models/chronic_model.sav',
            'liver_disease': 'models/liver_model.sav',
            'hepatitis': 'models/hepititisc_model.sav'
        }
        
        for disease, model_path in model_files.items():
            try:
                if os.path.exists(model_path):
                    self.traditional_models[disease] = joblib.load(model_path)
                    print(f"✅ Loaded traditional model for {disease}")
                else:
                    print(f"⚠️ Traditional model not found for {disease}")
            except Exception as e:
                print(f"❌ Error loading traditional model for {disease}: {e}")
    
    def create_stacking_ensemble(self, disease_name, X, y):
        """Create a stacking ensemble combining multiple models"""
        print(f"Creating stacking ensemble for {disease_name}...")
        
        # Base models
        base_models = []
        
        # Add traditional model if available
        if disease_name in self.traditional_models:
            base_models.append(('traditional', self.traditional_models[disease_name]))
        
        # Train and add deep learning model
        deep_model, _, _ = self.deep_predictor.train_deep_model(disease_name, X, y)
        
        # Create a wrapper for the deep model to work with sklearn
        class DeepModelWrapper:
            def __init__(self, model, scaler):
                self.model = model
                self.scaler = scaler
            
            def predict(self, X):
                X_scaled = self.scaler.transform(X)
                pred_probs = self.model.predict(X_scaled)
                return (pred_probs > 0.5).astype(int).flatten()
            
            def predict_proba(self, X):
                X_scaled = self.scaler.transform(X)
                pred_probs = self.model.predict(X_scaled).flatten()
                return np.column_stack([1 - pred_probs, pred_probs])
        
        deep_wrapper = DeepModelWrapper(deep_model, self.deep_predictor.scalers[disease_name])
        base_models.append(('deep', deep_wrapper))
        
        # Meta-learner
        meta_learner = LogisticRegression(random_state=42)
        
        # Create stacking classifier
        stacking_clf = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=5,
            stack_method='predict_proba'
        )
        
        return stacking_clf
    
    def create_weighted_ensemble(self, disease_name):
        """Create a weighted ensemble based on model performance"""
        config = self.ensemble_configs.get(disease_name, self.ensemble_configs['diabetes'])
        
        class WeightedEnsemble:
            def __init__(self, traditional_model, deep_model, deep_scaler, weights):
                self.traditional_model = traditional_model
                self.deep_model = deep_model
                self.deep_scaler = deep_scaler
                self.weights = weights
            
            def predict(self, X):
                predictions = []
                
                # Traditional model prediction
                if self.traditional_model is not None:
                    trad_pred = self.traditional_model.predict_proba(X)[:, 1]
                    predictions.append(trad_pred * self.weights['traditional'])
                
                # Deep model prediction
                if self.deep_model is not None:
                    X_scaled = self.deep_scaler.transform(X)
                    deep_pred = self.deep_model.predict(X_scaled).flatten()
                    predictions.append(deep_pred * self.weights['deep'])
                
                # Combine predictions
                if predictions:
                    final_pred = np.sum(predictions, axis=0)
                    return (final_pred > 0.5).astype(int)
                else:
                    return np.zeros(X.shape[0])
            
            def predict_proba(self, X):
                predictions = []
                
                # Traditional model prediction
                if self.traditional_model is not None:
                    trad_pred = self.traditional_model.predict_proba(X)[:, 1]
                    predictions.append(trad_pred * self.weights['traditional'])
                
                # Deep model prediction
                if self.deep_model is not None:
                    X_scaled = self.deep_scaler.transform(X)
                    deep_pred = self.deep_model.predict(X_scaled).flatten()
                    predictions.append(deep_pred * self.weights['deep'])
                
                # Combine predictions
                if predictions:
                    final_pred = np.sum(predictions, axis=0)
                    return np.column_stack([1 - final_pred, final_pred])
                else:
                    return np.column_stack([np.ones(X.shape[0]), np.zeros(X.shape[0])])
        
        weights = {
            'traditional': config['traditional_weight'],
            'deep': config['deep_weight']
        }
        
        traditional_model = self.traditional_models.get(disease_name)
        deep_model = self.deep_predictor.models.get(disease_name)
        deep_scaler = self.deep_predictor.scalers.get(disease_name)
        
        return WeightedEnsemble(traditional_model, deep_model, deep_scaler, weights)
    
    def predict_with_ensemble(self, disease_name, X, include_uncertainty=True):
        """Make predictions using ensemble approach"""
        if disease_name not in self.ensemble_models:
            # Try to load the ensemble model first
            if not self.load_ensemble_model(disease_name):
                # If ensemble model is not available, try to create one
                if not self.create_ensemble_model(disease_name):
                    raise ValueError(f"Ensemble model for {disease_name} not found and could not be created")

        ensemble_model = self.ensemble_models[disease_name]
        config = self.ensemble_configs.get(disease_name, self.ensemble_configs['diabetes'])

        # Validate that ensemble_model has the required methods
        if not hasattr(ensemble_model, 'predict_proba'):
            raise ValueError(f"Invalid ensemble model for {disease_name}. Expected model with predict_proba method.")
        
        # Get ensemble prediction
        ensemble_pred_proba = ensemble_model.predict_proba(X)
        ensemble_pred = ensemble_model.predict(X)
        
        results = {
            'prediction': ensemble_pred,
            'probability': ensemble_pred_proba[:, 1],
            'confidence_level': 'high' if np.max(ensemble_pred_proba, axis=1)[0] > config['confidence_threshold'] else 'medium'
        }
        
        if include_uncertainty:
            # Calculate prediction uncertainty using multiple approaches
            individual_predictions = []
            
            # Traditional model
            if disease_name in self.traditional_models:
                trad_pred = self.traditional_models[disease_name].predict_proba(X)[:, 1]
                individual_predictions.append(trad_pred)
            
            # Deep model with Monte Carlo dropout
            if disease_name in self.deep_predictor.models:
                deep_results = self.deep_predictor.predict_with_confidence(disease_name, X)
                individual_predictions.append(deep_results['probability'])
                results['uncertainty'] = deep_results['uncertainty']
                results['confidence_interval'] = deep_results['confidence_interval']
            
            # Calculate disagreement between models
            if len(individual_predictions) > 1:
                predictions_array = np.array(individual_predictions)
                disagreement = np.std(predictions_array, axis=0)
                results['model_disagreement'] = disagreement
                
                # Adjust confidence based on disagreement
                if np.mean(disagreement) > 0.2:
                    results['confidence_level'] = 'low'
                elif np.mean(disagreement) > 0.1:
                    results['confidence_level'] = 'medium'
        
        return results
    
    def create_multimodal_prediction(self, disease_name, tabular_data, image_data=None):
        """Create prediction combining tabular and image data"""
        results = {
            'tabular_prediction': None,
            'image_prediction': None,
            'combined_prediction': None,
            'confidence_scores': {}
        }
        
        # Tabular data prediction
        if tabular_data is not None:
            tabular_results = self.predict_with_ensemble(disease_name, tabular_data)
            results['tabular_prediction'] = tabular_results
            results['confidence_scores']['tabular'] = tabular_results.get('probability', [0])[0]
        
        # Image data prediction (if applicable)
        if image_data is not None and disease_name in ['heart_disease', 'chronic_kidney']:
            # Map disease to image type
            image_type_mapping = {
                'heart_disease': 'chest_xray',
                'chronic_kidney': 'retinal_scan'  # For diabetic complications
            }
            
            if disease_name in image_type_mapping:
                image_type = image_type_mapping[disease_name]
                try:
                    image_results = self.image_analyzer.analyze_image(image_type, image_data)
                    results['image_prediction'] = image_results
                    results['confidence_scores']['image'] = image_results.get('confidence', 0)
                except Exception as e:
                    print(f"Error in image analysis: {e}")
        
        # Combine predictions if both are available
        if results['tabular_prediction'] and results['image_prediction']:
            tabular_prob = results['confidence_scores']['tabular']
            image_prob = results['confidence_scores']['image']
            
            # Weighted combination (favor tabular data for now)
            combined_prob = 0.7 * tabular_prob + 0.3 * image_prob
            combined_pred = 1 if combined_prob > 0.5 else 0
            
            results['combined_prediction'] = {
                'prediction': combined_pred,
                'probability': combined_prob,
                'confidence_level': 'high' if combined_prob > 0.8 or combined_prob < 0.2 else 'medium'
            }
        
        return results
    
    def evaluate_ensemble_performance(self, disease_name, X_test, y_test):
        """Evaluate ensemble model performance"""
        if disease_name not in self.ensemble_models:
            raise ValueError(f"Ensemble model for {disease_name} not found")
        
        ensemble_model = self.ensemble_models[disease_name]
        
        # Make predictions
        y_pred = ensemble_model.predict(X_test)
        y_pred_proba = ensemble_model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Compare with individual models
        individual_performances = {}
        
        # Traditional model performance
        if disease_name in self.traditional_models:
            trad_pred = self.traditional_models[disease_name].predict(X_test)
            individual_performances['traditional'] = {
                'accuracy': accuracy_score(y_test, trad_pred),
                'f1_score': f1_score(y_test, trad_pred, average='weighted')
            }
        
        # Deep model performance
        if disease_name in self.deep_predictor.models:
            deep_results = self.deep_predictor.predict_with_confidence(disease_name, X_test)
            individual_performances['deep'] = {
                'accuracy': accuracy_score(y_test, deep_results['prediction']),
                'f1_score': f1_score(y_test, deep_results['prediction'], average='weighted')
            }
        
        results = {
            'ensemble_performance': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            },
            'individual_performances': individual_performances,
            'improvement': {}
        }
        
        # Calculate improvement over individual models
        for model_type, perf in individual_performances.items():
            results['improvement'][model_type] = {
                'accuracy_improvement': accuracy - perf['accuracy'],
                'f1_improvement': f1 - perf['f1_score']
            }
        
        return results
    
    def save_ensemble_model(self, disease_name, model_path=None):
        """Save ensemble model"""
        if disease_name not in self.ensemble_models:
            raise ValueError(f"Ensemble model for {disease_name} not found")
        
        if model_path is None:
            model_path = f"models/ensemble/{disease_name}_ensemble_model.pkl"
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.ensemble_models[disease_name], model_path)
        
        return model_path
    
    def load_ensemble_model(self, disease_name, model_path=None):
        """Load ensemble model"""
        if model_path is None:
            model_path = f"models/ensemble/{disease_name}_ensemble_model.pkl"

        try:
            self.ensemble_models[disease_name] = joblib.load(model_path)
            return True
        except Exception as e:
            print(f"Error loading ensemble model for {disease_name}: {e}")
            return False

    def create_ensemble_model(self, disease_name):
        """Create a simple ensemble model if not available"""
        try:
            # Try to load traditional model
            traditional_model = self.traditional_models.get(disease_name)

            if traditional_model is None:
                print(f"No traditional model available for {disease_name}")
                return False

            # Create a simple wrapper that just uses the traditional model
            class SimpleEnsemble:
                def __init__(self, traditional_model):
                    self.traditional_model = traditional_model

                def predict(self, X):
                    return self.traditional_model.predict(X)

                def predict_proba(self, X):
                    if hasattr(self.traditional_model, 'predict_proba'):
                        return self.traditional_model.predict_proba(X)
                    else:
                        # Fallback for models without predict_proba
                        predictions = self.traditional_model.predict(X)
                        probs = np.zeros((len(predictions), 2))
                        probs[predictions == 0, 0] = 0.8
                        probs[predictions == 0, 1] = 0.2
                        probs[predictions == 1, 0] = 0.2
                        probs[predictions == 1, 1] = 0.8
                        return probs

            self.ensemble_models[disease_name] = SimpleEnsemble(traditional_model)
            print(f"Created simple ensemble model for {disease_name}")
            return True

        except Exception as e:
            print(f"Error creating ensemble model for {disease_name}: {e}")
            return False
