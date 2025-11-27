#!/usr/bin/env python3
"""
🔍 EXPLAINABLE AI MODULE
=======================
Comprehensive AI explainability for medical predictions using SHAP and custom analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Try importing SHAP
try:
    import shap
    SHAP_AVAILABLE = True
    print("✅ SHAP library available for explainable AI")
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available. Installing...")
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
        import shap
        SHAP_AVAILABLE = True
        print("✅ SHAP installed successfully")
    except:
        SHAP_AVAILABLE = False
        print("❌ Could not install SHAP")

class MedicalExplainableAI:
    """
    Advanced Explainable AI for Medical Predictions
    """
    
    def __init__(self):
        self.feature_explanations = self._get_feature_explanations()
        self.precautions_database = self._get_precautions_database()
        self.risk_thresholds = self._get_risk_thresholds()
    
    def _get_feature_explanations(self):
        """Comprehensive medical parameter explanations"""
        return {
            # Diabetes Features
            'glucose': {
                'name': 'Blood Glucose Level',
                'description': 'Amount of sugar (glucose) in your blood',
                'normal_range': '70-100 mg/dL (fasting)',
                'high_risk': '>126 mg/dL indicates diabetes risk',
                'what_it_means': 'High glucose levels indicate your body cannot process sugar properly',
                'icon': '🍯'
            },
            'blood_pressure': {
                'name': 'Blood Pressure (Systolic)',
                'description': 'Pressure in arteries when heart beats',
                'normal_range': '90-120 mmHg',
                'high_risk': '>140 mmHg indicates hypertension',
                'what_it_means': 'High BP puts strain on heart and blood vessels',
                'icon': '💓'
            },
            'insulin': {
                'name': 'Insulin Level',
                'description': 'Hormone that regulates blood sugar',
                'normal_range': '16-166 pmol/L',
                'high_risk': 'Very high or very low levels are concerning',
                'what_it_means': 'Abnormal insulin levels indicate metabolic dysfunction',
                'icon': '💉'
            },
            'bmi': {
                'name': 'Body Mass Index',
                'description': 'Measure of body fat based on height and weight',
                'normal_range': '18.5-24.9',
                'high_risk': '>30 indicates obesity',
                'what_it_means': 'Higher BMI increases risk of various diseases',
                'icon': '⚖️'
            },
            'age': {
                'name': 'Age',
                'description': 'Your current age in years',
                'normal_range': 'N/A',
                'high_risk': 'Risk increases with age',
                'what_it_means': 'Older age is associated with higher disease risk',
                'icon': '📅'
            },
            'pregnancies': {
                'name': 'Number of Pregnancies',
                'description': 'Total number of pregnancies',
                'normal_range': 'N/A',
                'high_risk': 'Multiple pregnancies may increase diabetes risk',
                'what_it_means': 'Pregnancy can stress the body\'s glucose regulation',
                'icon': '🤱'
            },
            'skin_thickness': {
                'name': 'Skin Thickness',
                'description': 'Triceps skin fold thickness (mm)',
                'normal_range': '10-40 mm',
                'high_risk': 'Very high values may indicate insulin resistance',
                'what_it_means': 'Related to body fat percentage and insulin sensitivity',
                'icon': '📏'
            },
            'pedigree': {
                'name': 'Diabetes Pedigree Function',
                'description': 'Genetic likelihood of diabetes based on family history',
                'normal_range': '0.078-2.42',
                'high_risk': '>1.0 indicates strong genetic predisposition',
                'what_it_means': 'Higher values mean stronger family history of diabetes',
                'icon': '🧬'
            },
            
            # Heart Disease Features
            'chest_pain': {
                'name': 'Chest Pain Type',
                'description': 'Type of chest pain experienced',
                'normal_range': '0: Typical angina, 1: Atypical angina, 2: Non-anginal, 3: Asymptomatic',
                'high_risk': 'Typical angina (0) is most concerning',
                'what_it_means': 'Different chest pain types indicate different heart conditions',
                'icon': '💔'
            },
            'cholesterol': {
                'name': 'Cholesterol Level',
                'description': 'Total cholesterol in blood (mg/dL)',
                'normal_range': '<200 mg/dL',
                'high_risk': '>240 mg/dL is high risk',
                'what_it_means': 'High cholesterol clogs arteries and increases heart attack risk',
                'icon': '🧪'
            },
            'fasting_sugar': {
                'name': 'Fasting Blood Sugar',
                'description': 'Blood sugar after 12-hour fast',
                'normal_range': '<100 mg/dL',
                'high_risk': '>126 mg/dL indicates diabetes',
                'what_it_means': 'High fasting sugar indicates metabolic problems',
                'icon': '🍬'
            },
            'ecg': {
                'name': 'ECG Results',
                'description': 'Electrocardiogram results',
                'normal_range': '0: Normal, 1: ST-T abnormality, 2: Left ventricular hypertrophy',
                'high_risk': 'Abnormal results (1,2) indicate heart problems',
                'what_it_means': 'ECG shows electrical activity and structure of heart',
                'icon': '📈'
            },
            'max_heart_rate': {
                'name': 'Maximum Heart Rate',
                'description': 'Highest heart rate achieved during exercise',
                'normal_range': '220 - age (approximate)',
                'high_risk': 'Very low max HR may indicate heart problems',
                'what_it_means': 'Lower max heart rate may indicate reduced cardiac fitness',
                'icon': '💗'
            },
            'angina': {
                'name': 'Exercise Induced Angina',
                'description': 'Chest pain during physical activity',
                'normal_range': '0: No, 1: Yes',
                'high_risk': 'Yes (1) indicates reduced blood flow to heart',
                'what_it_means': 'Exercise angina suggests coronary artery blockage',
                'icon': '🏃‍♂️'
            },
            'depression': {
                'name': 'ST Depression',
                'description': 'ECG change during exercise',
                'normal_range': '0-3',
                'high_risk': '>2 indicates significant heart problems',
                'what_it_means': 'ST depression shows reduced blood flow to heart muscle',
                'icon': '📉'
            },
            
            # Parkinson's Features (Voice Analysis)
            'fo': {
                'name': 'Fundamental Frequency',
                'description': 'Average vocal fundamental frequency',
                'normal_range': '88-260 Hz',
                'high_risk': 'Extreme values may indicate voice changes',
                'what_it_means': 'Changes in voice pitch can indicate neurological issues',
                'icon': '🗣️'
            },
            'jitter_percent': {
                'name': 'Jitter Percentage',
                'description': 'Variation in vocal frequency',
                'normal_range': '<1%',
                'high_risk': '>3% indicates voice instability',
                'what_it_means': 'High jitter shows vocal cord irregularities',
                'icon': '🎵'
            },
            'shimmer': {
                'name': 'Shimmer',
                'description': 'Variation in vocal amplitude',
                'normal_range': '<3%',
                'high_risk': '>11% indicates voice problems',
                'what_it_means': 'High shimmer shows vocal strength variations',
                'icon': '🔊'
            },
            'nhr': {
                'name': 'Noise-to-Harmonics Ratio',
                'description': 'Ratio of noise to harmonic components',
                'normal_range': '<0.2',
                'high_risk': '>0.4 indicates voice quality issues',
                'what_it_means': 'High NHR shows breathiness and voice deterioration',
                'icon': '🎤'
            },
            'hnr': {
                'name': 'Harmonics-to-Noise Ratio',
                'description': 'Ratio of harmonic to noise components',
                'normal_range': '>20 dB',
                'high_risk': '<15 dB indicates voice problems',
                'what_it_means': 'Low HNR shows reduced voice quality',
                'icon': '📊'
            },
            
            # Liver Disease Features  
            'total_bilirubin': {
                'name': 'Total Bilirubin',
                'description': 'Waste product from red blood cell breakdown',
                'normal_range': '0.3-1.2 mg/dL',
                'high_risk': '>2.0 mg/dL indicates liver problems',
                'what_it_means': 'High bilirubin causes jaundice and indicates liver dysfunction',
                'icon': '🟡'
            },
            'alkaline_phosphotase': {
                'name': 'Alkaline Phosphatase',
                'description': 'Enzyme found in liver and bones',
                'normal_range': '44-147 IU/L',
                'high_risk': '>300 IU/L indicates liver damage',
                'what_it_means': 'High levels indicate liver or bone disease',
                'icon': '🧬'
            },
            'alamine_aminotransferase': {
                'name': 'ALT (Alanine Aminotransferase)',
                'description': 'Liver enzyme indicating liver health',
                'normal_range': '7-56 IU/L',
                'high_risk': '>100 IU/L indicates liver damage',
                'what_it_means': 'High ALT shows liver cell damage',
                'icon': '🔬'
            },
            'aspartate_aminotransferase': {
                'name': 'AST (Aspartate Aminotransferase)',
                'description': 'Enzyme found in liver and other organs',
                'normal_range': '10-40 IU/L',
                'high_risk': '>100 IU/L indicates organ damage',
                'what_it_means': 'High AST indicates liver or heart muscle damage',
                'icon': '⚗️'
            },
            'total_proteins': {
                'name': 'Total Proteins',
                'description': 'All proteins in blood serum',
                'normal_range': '6.0-8.3 g/dL',
                'high_risk': '<6.0 or >8.5 g/dL is concerning',
                'what_it_means': 'Abnormal protein levels indicate liver or kidney problems',
                'icon': '🥩'
            },
            'albumin': {
                'name': 'Albumin',
                'description': 'Main protein made by liver',
                'normal_range': '3.5-5.0 g/dL',
                'high_risk': '<3.5 g/dL indicates liver problems',
                'what_it_means': 'Low albumin shows reduced liver function',
                'icon': '🥛'
            },
            
            # Kidney Disease Features
            'blood_urea': {
                'name': 'Blood Urea',
                'description': 'Waste product filtered by kidneys',
                'normal_range': '7-20 mg/dL',
                'high_risk': '>50 mg/dL indicates kidney problems',
                'what_it_means': 'High urea shows kidneys not filtering waste properly',
                'icon': '🩸'
            },
            'serum_creatinine': {
                'name': 'Serum Creatinine',
                'description': 'Waste product from muscle metabolism',
                'normal_range': '0.6-1.2 mg/dL',
                'high_risk': '>2.0 mg/dL indicates kidney damage',
                'what_it_means': 'High creatinine shows reduced kidney function',
                'icon': '💪'
            },
            'hemoglobin': {
                'name': 'Hemoglobin',
                'description': 'Protein in red blood cells that carries oxygen',
                'normal_range': '12-15.5 g/dL (women), 14-17.5 g/dL (men)',
                'high_risk': '<10 g/dL indicates anemia',
                'what_it_means': 'Low hemoglobin causes fatigue and weakness',
                'icon': '🔴'
            },
            'hypertension': {
                'name': 'Hypertension',
                'description': 'High blood pressure condition',
                'normal_range': '0: No, 1: Yes',
                'high_risk': 'Yes (1) increases risk of complications',
                'what_it_means': 'High blood pressure damages blood vessels and organs',
                'icon': '⚡'
            }
        }
    
    def _get_precautions_database(self):
        """Comprehensive precautions database"""
        return {
            'diabetes': {
                'high_glucose': [
                    "🍎 Follow a low-sugar, low-carb diet",
                    "🚶‍♂️ Exercise regularly (30 min daily walking)",
                    "💊 Take prescribed diabetes medications",
                    "🩺 Monitor blood sugar levels daily",
                    "⚖️ Maintain healthy weight",
                    "🚭 Avoid smoking and excessive alcohol"
                ],
                'high_bmi': [
                    "🥗 Adopt a balanced, calorie-controlled diet",
                    "🏃‍♀️ Increase physical activity gradually",
                    "💧 Drink plenty of water",
                    "😴 Get adequate sleep (7-8 hours)",
                    "🧘‍♀️ Practice stress management",
                    "👨‍⚕️ Consult a nutritionist"
                ],
                'high_blood_pressure': [
                    "🧂 Reduce sodium intake (<2300mg/day)",
                    "🥬 Eat potassium-rich foods (bananas, leafy greens)",
                    "🏊‍♂️ Regular cardiovascular exercise",
                    "🚫 Limit caffeine and alcohol",
                    "😌 Practice relaxation techniques",
                    "💊 Take BP medications as prescribed"
                ],
                'genetic_risk': [
                    "🧬 Regular genetic counseling",
                    "🔍 More frequent health screenings",
                    "📚 Learn about family medical history",
                    "🥗 Preventive lifestyle changes",
                    "👨‍⚕️ Discuss preventive medications with doctor"
                ]
            },
            'heart_disease': {
                'high_cholesterol': [
                    "🐟 Eat omega-3 rich foods (fish, nuts)",
                    "🚫 Avoid trans fats and saturated fats",
                    "🌾 Increase fiber intake (oats, beans)",
                    "🥑 Include healthy fats (avocado, olive oil)",
                    "💊 Consider cholesterol medications if prescribed",
                    "🏃‍♂️ Regular aerobic exercise"
                ],
                'chest_pain': [
                    "🚨 Seek immediate medical attention for chest pain",
                    "💊 Keep nitroglycerin if prescribed",
                    "📱 Know when to call emergency services",
                    "🚫 Avoid strenuous activities until cleared",
                    "🧘‍♂️ Learn stress management techniques"
                ],
                'high_blood_pressure': [
                    "🧂 Follow DASH diet (low sodium)",
                    "🏃‍♀️ Regular moderate exercise",
                    "⚖️ Maintain healthy weight",
                    "🚭 Quit smoking immediately",
                    "🍷 Limit alcohol consumption",
                    "💊 Take BP medications consistently"
                ],
                'exercise_limitations': [
                    "👨‍⚕️ Get cardiac clearance before exercising",
                    "🚶‍♂️ Start with low-intensity activities",
                    "📈 Gradually increase exercise intensity",
                    "💓 Monitor heart rate during exercise",
                    "🛑 Stop exercise if chest pain occurs"
                ]
            },
            'parkinsons': {
                'voice_changes': [
                    "🗣️ Practice voice exercises daily",
                    "👨‍⚕️ Consider speech therapy",
                    "🎵 Singing can help maintain voice strength",
                    "💧 Stay hydrated for vocal cord health",
                    "📢 Speak slowly and clearly"
                ],
                'motor_symptoms': [
                    "🤸‍♂️ Regular physical therapy",
                    "🏃‍♂️ Exercise to maintain mobility",
                    "🎯 Practice balance exercises",
                    "💊 Take medications on schedule",
                    "🏠 Make home safety modifications"
                ],
                'general_care': [
                    "💊 Consistent medication timing",
                    "🥗 Balanced nutrition",
                    "😴 Maintain good sleep hygiene",
                    "🧘‍♂️ Stress reduction techniques",
                    "👥 Stay socially active"
                ]
            },
            'liver_disease': {
                'high_enzymes': [
                    "🍷 Completely avoid alcohol",
                    "💊 Avoid unnecessary medications",
                    "🥗 Eat liver-friendly foods",
                    "⚖️ Maintain healthy weight",
                    "💧 Stay well hydrated"
                ],
                'high_bilirubin': [
                    "👨‍⚕️ Immediate medical evaluation needed",
                    "🟡 Monitor for jaundice symptoms",
                    "🚫 Avoid liver-toxic substances",
                    "🥗 Low-fat, easy-to-digest diet"
                ],
                'general_liver_health': [
                    "💉 Get vaccinated for Hepatitis A & B",
                    "🧼 Practice good hygiene",
                    "🚫 Avoid sharing needles or razors",
                    "🥗 Eat antioxidant-rich foods",
                    "☕ Moderate coffee consumption may help"
                ]
            },
            'hepatitis': {
                'liver_inflammation': [
                    "🍷 Complete alcohol abstinence",
                    "💊 Take antiviral medications as prescribed",
                    "🥗 Nutritious, balanced diet",
                    "😴 Adequate rest and sleep",
                    "🚫 Avoid hepatotoxic substances"
                ],
                'prevention': [
                    "💉 Complete vaccination series",
                    "🧼 Frequent hand washing",
                    "🚫 Don't share personal items",
                    "🍽️ Practice food safety",
                    "🩸 Safe blood transfusion practices"
                ]
            },
            'chronic_kidney': {
                'high_creatinine': [
                    "💧 Stay adequately hydrated",
                    "🧂 Limit sodium intake",
                    "🥩 Moderate protein consumption",
                    "💊 Monitor medication dosages",
                    "📊 Regular kidney function tests"
                ],
                'high_blood_pressure': [
                    "💊 Strict BP medication compliance",
                    "🧂 Very low sodium diet",
                    "⚖️ Weight management",
                    "🏃‍♂️ Regular exercise as tolerated"
                ],
                'general_kidney_health': [
                    "🍬 Control diabetes strictly",
                    "🚭 Quit smoking",
                    "💊 Avoid NSAIDs",
                    "🥗 Kidney-friendly diet",
                    "👨‍⚕️ Regular nephrology follow-up"
                ]
            }
        }
    
    def _get_risk_thresholds(self):
        """Risk thresholds for different parameters"""
        return {
            'diabetes': {
                'glucose': {'low': 100, 'moderate': 126, 'high': 200},
                'bmi': {'low': 25, 'moderate': 30, 'high': 35},
                'blood_pressure': {'low': 120, 'moderate': 140, 'high': 180},
                'age': {'low': 45, 'moderate': 65, 'high': 75}
            },
            'heart_disease': {
                'cholesterol': {'low': 200, 'moderate': 240, 'high': 300},
                'blood_pressure': {'low': 120, 'moderate': 140, 'high': 180},
                'age': {'low': 40, 'moderate': 55, 'high': 65}
            },
            'liver_disease': {
                'total_bilirubin': {'low': 1.2, 'moderate': 2.0, 'high': 5.0},
                'alamine_aminotransferase': {'low': 56, 'moderate': 100, 'high': 200},
                'aspartate_aminotransferase': {'low': 40, 'moderate': 100, 'high': 200}
            }
        }
    
    def explain_ml_prediction(self, model, X_input, feature_names, disease_name, model_type="ml"):
        """
        Generate comprehensive explanation for ML/DL predictions
        """
        explanation_data = {
            'disease': disease_name,
            'model_type': model_type,
            'prediction_confidence': 0.0,
            'feature_importance': {},
            'risk_factors': [],
            'precautions': [],
            'parameter_analysis': {}
        }
        
        try:
            # Get prediction and confidence
            if hasattr(model, 'predict_proba'):
                prediction = model.predict(X_input)[0]
                probabilities = model.predict_proba(X_input)[0]
                confidence = max(probabilities)
            else:
                prediction = model.predict(X_input)[0]
                confidence = 0.8  # Default confidence for models without probability
            
            explanation_data['prediction'] = int(prediction)
            explanation_data['prediction_confidence'] = float(confidence)
            
            # SHAP Analysis (if available)
            if SHAP_AVAILABLE and hasattr(model, 'predict'):
                try:
                    explainer = shap.Explainer(model, X_input)
                    shap_values = explainer(X_input)
                    
                    # Get feature importance from SHAP
                    if hasattr(shap_values, 'values'):
                        shap_importance = np.abs(shap_values.values[0])
                        for i, feature in enumerate(feature_names):
                            explanation_data['feature_importance'][feature] = float(shap_importance[i])
                except Exception as e:
                    print(f"SHAP analysis failed: {e}")
                    # Fallback to model-based feature importance
                    self._get_fallback_importance(model, X_input, feature_names, explanation_data)
            else:
                # Fallback feature importance
                self._get_fallback_importance(model, X_input, feature_names, explanation_data)
            
            # Analyze individual parameters
            explanation_data['parameter_analysis'] = self._analyze_parameters(
                X_input[0], feature_names, disease_name
            )
            
            # Get risk factors and precautions
            explanation_data['risk_factors'] = self._identify_risk_factors(
                X_input[0], feature_names, disease_name, explanation_data['feature_importance']
            )
            
            explanation_data['precautions'] = self._get_relevant_precautions(
                disease_name, explanation_data['risk_factors']
            )
            
        except Exception as e:
            print(f"Error in explanation generation: {e}")
            explanation_data['error'] = str(e)
        
        return explanation_data
    
    def _get_fallback_importance(self, model, X_input, feature_names, explanation_data):
        """Fallback method for feature importance when SHAP is not available"""
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                for i, feature in enumerate(feature_names):
                    explanation_data['feature_importance'][feature] = float(importances[i])
            elif hasattr(model, 'coef_'):
                # Linear models
                coefficients = np.abs(model.coef_[0])
                for i, feature in enumerate(feature_names):
                    explanation_data['feature_importance'][feature] = float(coefficients[i])
            else:
                # Default equal importance
                for feature in feature_names:
                    explanation_data['feature_importance'][feature] = 1.0 / len(feature_names)
        except Exception as e:
            print(f"Fallback importance calculation failed: {e}")
            # Set equal importance as last resort
            for feature in feature_names:
                explanation_data['feature_importance'][feature] = 1.0 / len(feature_names)
    
    def _analyze_parameters(self, input_values, feature_names, disease_name):
        """Analyze individual parameter values"""
        analysis = {}
        thresholds = self._get_risk_thresholds().get(disease_name, {})
        
        for i, feature in enumerate(feature_names):
            value = input_values[i]
            feature_info = self.feature_explanations.get(feature, {})
            feature_thresholds = thresholds.get(feature, {})
            
            # Determine risk level
            if feature_thresholds:
                if value <= feature_thresholds.get('low', 0):
                    risk_level = 'Low'
                    risk_color = 'green'
                elif value <= feature_thresholds.get('moderate', float('inf')):
                    risk_level = 'Moderate'
                    risk_color = 'orange'
                else:
                    risk_level = 'High'
                    risk_color = 'red'
            else:
                risk_level = 'Unknown'
                risk_color = 'gray'
            
            analysis[feature] = {
                'value': float(value),
                'risk_level': risk_level,
                'risk_color': risk_color,
                'explanation': feature_info.get('description', 'Parameter description not available'),
                'normal_range': feature_info.get('normal_range', 'Range not specified'),
                'what_it_means': feature_info.get('what_it_means', 'Significance not specified'),
                'icon': feature_info.get('icon', '📊')
            }
        
        return analysis
    
    def _identify_risk_factors(self, input_values, feature_names, disease_name, importance_scores):
        """Identify the most significant risk factors"""
        risk_factors = []
        
        # Sort features by importance
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top risk factors (top 5 or features with importance > 0.1)
        for feature, importance in sorted_features[:5]:
            if importance > 0.1:  # Only include significant features
                feature_idx = feature_names.index(feature) if feature in feature_names else -1
                if feature_idx >= 0:
                    value = input_values[feature_idx]
                    feature_info = self.feature_explanations.get(feature, {})
                    
                    risk_factors.append({
                        'parameter': feature,
                        'value': float(value),
                        'importance': float(importance),
                        'explanation': feature_info.get('what_it_means', 'This parameter contributes to disease risk'),
                        'icon': feature_info.get('icon', '⚠️')
                    })
        
        return risk_factors
    
    def _get_relevant_precautions(self, disease_name, risk_factors):
        """Get relevant precautions based on identified risk factors"""
        precautions = []
        disease_precautions = self.precautions_database.get(disease_name, {})
        
        # Get general precautions
        if 'general_care' in disease_precautions:
            precautions.extend(disease_precautions['general_care'])
        
        # Get specific precautions based on risk factors
        for risk_factor in risk_factors:
            parameter = risk_factor['parameter']
            
            # Map parameters to precaution categories
            if parameter in ['glucose', 'fasting_sugar']:
                precautions.extend(disease_precautions.get('high_glucose', []))
            elif parameter in ['bmi']:
                precautions.extend(disease_precautions.get('high_bmi', []))
            elif parameter in ['blood_pressure']:
                precautions.extend(disease_precautions.get('high_blood_pressure', []))
            elif parameter in ['cholesterol']:
                precautions.extend(disease_precautions.get('high_cholesterol', []))
            elif parameter in ['chest_pain']:
                precautions.extend(disease_precautions.get('chest_pain', []))
            elif parameter in ['total_bilirubin']:
                precautions.extend(disease_precautions.get('high_bilirubin', []))
            elif parameter in ['alamine_aminotransferase', 'aspartate_aminotransferase']:
                precautions.extend(disease_precautions.get('high_enzymes', []))
            elif parameter in ['serum_creatinine', 'blood_urea']:
                precautions.extend(disease_precautions.get('high_creatinine', []))
        
        # Remove duplicates while preserving order
        unique_precautions = []
        for precaution in precautions:
            if precaution not in unique_precautions:
                unique_precautions.append(precaution)
        
        return unique_precautions[:10]  # Limit to top 10 precautions
    
    def visualize_feature_importance(self, importance_scores, disease_name):
        """Create interactive visualization of feature importance"""
        if not importance_scores:
            return None
        
        # Prepare data for visualization
        features = list(importance_scores.keys())
        importances = list(importance_scores.values())
        
        # Get feature explanations
        feature_labels = []
        icons = []
        for feature in features:
            info = self.feature_explanations.get(feature, {})
            label = info.get('name', feature.replace('_', ' ').title())
            icon = info.get('icon', '📊')
            feature_labels.append(f"{icon} {label}")
            icons.append(icon)
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        # Sort by importance
        sorted_data = sorted(zip(feature_labels, importances, features), key=lambda x: x[1], reverse=True)
        sorted_labels, sorted_importances, sorted_features = zip(*sorted_data)
        
        # Color scheme based on importance
        colors = ['#FF6B6B' if imp > 0.2 else '#4ECDC4' if imp > 0.1 else '#45B7D1' 
                  for imp in sorted_importances]
        
        fig.add_trace(go.Bar(
            y=sorted_labels,
            x=sorted_importances,
            orientation='h',
            marker=dict(color=colors),
            text=[f'{imp:.3f}' for imp in sorted_importances],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'🔍 Feature Importance Analysis - {disease_name.title()}',
            xaxis_title='Importance Score',
            yaxis_title='Parameters',
            height=max(400, len(features) * 30),
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def create_risk_dashboard(self, parameter_analysis, disease_name):
        """Create comprehensive risk analysis dashboard"""
        if not parameter_analysis:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('📊 Parameter Risk Levels', '⚠️ High Risk Parameters', 
                          '✅ Normal Parameters', '📈 Risk Distribution'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Prepare data
        parameters = list(parameter_analysis.keys())
        risk_levels = [param['risk_level'] for param in parameter_analysis.values()]
        risk_colors_map = {'Low': 'green', 'Moderate': 'orange', 'High': 'red', 'Unknown': 'gray'}
        
        # 1. Risk levels bar chart
        risk_counts = {'Low': 0, 'Moderate': 0, 'High': 0, 'Unknown': 0}
        for level in risk_levels:
            risk_counts[level] += 1
        
        fig.add_trace(
            go.Bar(x=list(risk_counts.keys()), y=list(risk_counts.values()),
                   marker_color=['green', 'orange', 'red', 'gray'],
                   name='Risk Distribution'),
            row=1, col=1
        )
        
        # 2. High risk parameters
        high_risk_params = [param for param, data in parameter_analysis.items() 
                           if data['risk_level'] == 'High']
        if high_risk_params:
            high_risk_values = [parameter_analysis[param]['value'] for param in high_risk_params]
            fig.add_trace(
                go.Bar(x=high_risk_params, y=high_risk_values,
                       marker_color='red', name='High Risk Values'),
                row=1, col=2
            )
        
        # 3. Normal parameters
        normal_params = [param for param, data in parameter_analysis.items() 
                        if data['risk_level'] == 'Low']
        if normal_params:
            normal_values = [parameter_analysis[param]['value'] for param in normal_params]
            fig.add_trace(
                go.Bar(x=normal_params, y=normal_values,
                       marker_color='green', name='Normal Values'),
                row=2, col=1
            )
        
        # 4. Risk distribution pie chart
        fig.add_trace(
            go.Pie(labels=list(risk_counts.keys()), values=list(risk_counts.values()),
                   marker_colors=['green', 'orange', 'red', 'gray'],
                   name='Risk Distribution'),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'🏥 Comprehensive Risk Analysis - {disease_name.title()}',
            height=800,
            showlegend=False
        )
        
        return fig
    
    def generate_explanation_report(self, explanation_data):
        """Generate a comprehensive text report"""
        disease = explanation_data.get('disease', 'Unknown')
        prediction = explanation_data.get('prediction', 0)
        confidence = explanation_data.get('prediction_confidence', 0)
        risk_factors = explanation_data.get('risk_factors', [])
        precautions = explanation_data.get('precautions', [])
        
        report = f"""
# 🏥 Medical AI Analysis Report - {disease.title()}

## 📋 Prediction Summary
- **Result**: {'⚠️ Positive Risk Detected' if prediction else '✅ Low Risk'}
- **Confidence**: {confidence:.1%}
- **Analysis Type**: AI-Powered Medical Prediction

## 🔍 Top Risk Factors Identified

"""
        
        for i, factor in enumerate(risk_factors[:5], 1):
            report += f"""
### {i}. {factor['icon']} {factor['parameter'].replace('_', ' ').title()}
- **Current Value**: {factor['value']:.2f}
- **Impact Level**: {factor['importance']:.1%}
- **What this means**: {factor['explanation']}
"""
        
        if precautions:
            report += f"""
## 🛡️ Recommended Precautions

Based on your analysis, here are the most important steps to take:

"""
            for i, precaution in enumerate(precautions[:8], 1):
                report += f"{i}. {precaution}\n"
        
        report += f"""

## ⚠️ Important Medical Disclaimer

This analysis is provided by AI for informational purposes only and should not replace professional medical advice. Please consult with a qualified healthcare provider for proper diagnosis and treatment recommendations.

---
*Analysis generated by Advanced Medical AI System*
"""
        
        return report