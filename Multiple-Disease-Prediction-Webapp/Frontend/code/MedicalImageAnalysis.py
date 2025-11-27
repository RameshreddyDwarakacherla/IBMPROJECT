#!/usr/bin/env python3
"""
Medical Image Analysis Module
Advanced AI for medical image classification and analysis
"""

import numpy as np
import pandas as pd
import cv2
import os
import pickle
import json
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    print(f"TensorFlow not available for image analysis: {e}")
    TENSORFLOW_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class MedicalImageAnalyzer:
    """
    Advanced Medical Image Analysis using Deep Learning
    """
    
    def __init__(self):
        if not TENSORFLOW_AVAILABLE:
            print("Warning: TensorFlow not available. Image analysis features limited.")
            
        self.models = {}
        self.scalers = {}
        self.image_configs = {
            'chest_xray': {
                'input_shape': (224, 224, 3),
                'classes': ['Normal', 'Pneumonia'],
                'preprocessing': 'chest'
            },
            'brain_mri': {
                'input_shape': (224, 224, 3),
                'classes': ['Normal', 'Tumor'],
                'preprocessing': 'brain'
            },
            'skin_lesion': {
                'input_shape': (224, 224, 3),
                'classes': ['Benign', 'Malignant'],
                'preprocessing': 'skin'
            },
            'retinal_scan': {
                'input_shape': (224, 224, 3),
                'classes': ['Normal', 'Diabetic Retinopathy'],
                'preprocessing': 'retinal'
            }
        }
    
    def create_cnn_model(self, input_shape, num_classes=2):
        """Create CNN model for medical image classification"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for CNN models")
            
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fully Connected Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1 if num_classes == 2 else num_classes, 
                        activation='sigmoid' if num_classes == 2 else 'softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_medical_image(self, image_path, image_type='chest_xray'):
        """Preprocess medical image for analysis"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        config = self.image_configs.get(image_type, self.image_configs['chest_xray'])
        target_size = config['input_shape'][:2]
        
        # Load and resize image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        image = cv2.resize(image, target_size)
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Apply specific preprocessing based on image type
        if config['preprocessing'] == 'chest':
            # Enhance contrast for chest X-rays
            image = self.enhance_contrast(image)
        elif config['preprocessing'] == 'brain':
            # Apply brain-specific preprocessing
            image = self.enhance_brain_features(image)
        elif config['preprocessing'] == 'skin':
            # Skin lesion preprocessing
            image = self.enhance_skin_features(image)
        elif config['preprocessing'] == 'retinal':
            # Retinal scan preprocessing
            image = self.enhance_retinal_features(image)
        
        return np.expand_dims(image, axis=0)
    
    def enhance_contrast(self, image):
        """Enhance contrast for better feature visibility"""
        # Convert to LAB color space
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced.astype(np.float32) / 255.0
    
    def enhance_brain_features(self, image):
        """Enhance features specific to brain MRI"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur((image * 255).astype(np.uint8), (3, 3), 0)
        
        # Enhance edges
        edges = cv2.Canny(blurred, 50, 150)
        edges = np.expand_dims(edges, axis=2)
        edges = np.repeat(edges, 3, axis=2)
        
        # Combine original with edge information
        enhanced = image * 0.8 + (edges.astype(np.float32) / 255.0) * 0.2
        
        return np.clip(enhanced, 0, 1)
    
    def enhance_skin_features(self, image):
        """Enhance features for skin lesion analysis"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # Enhance saturation
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.2)
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return enhanced.astype(np.float32) / 255.0
    
    def enhance_retinal_features(self, image):
        """Enhance features for retinal scan analysis"""
        # Apply green channel enhancement (blood vessels are more visible in green)
        enhanced = image.copy()
        enhanced[:, :, 1] = np.clip(enhanced[:, :, 1] * 1.3, 0, 1)
        
        return enhanced
    
    def analyze_medical_image(self, image_path, image_type='chest_xray'):
        """Analyze medical image and provide diagnosis"""
        if not TENSORFLOW_AVAILABLE:
            return self.fallback_image_analysis(image_path, image_type)
            
        try:
            # Load model if not already loaded
            if image_type not in self.models:
                if not self.load_image_model(image_type):
                    return self.fallback_image_analysis(image_path, image_type)
            
            # Preprocess image
            processed_image = self.preprocess_medical_image(image_path, image_type)
            
            # Make prediction
            model = self.models[image_type]
            prediction = model.predict(processed_image, verbose=0)
            
            # Get class probabilities
            config = self.image_configs[image_type]
            classes = config['classes']
            
            if len(classes) == 2:
                prob = prediction[0][0]
                predicted_class = classes[1] if prob > 0.5 else classes[0]
                confidence = prob if prob > 0.5 else 1 - prob
            else:
                prob_idx = np.argmax(prediction[0])
                predicted_class = classes[prob_idx]
                confidence = prediction[0][prob_idx]
            
            # Generate analysis report
            analysis = {
                'prediction': predicted_class,
                'confidence': float(confidence),
                'all_probabilities': {
                    classes[i]: float(prediction[0][i] if len(classes) > 2 else 
                                    (prediction[0][0] if i == 1 else 1 - prediction[0][0]))
                    for i in range(len(classes))
                },
                'image_type': image_type,
                'model_used': 'Deep Learning CNN'
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error in image analysis: {e}")
            return self.fallback_image_analysis(image_path, image_type)
    
    def fallback_image_analysis(self, image_path, image_type):
        """Fallback analysis when deep learning is not available"""
        # Simple rule-based analysis
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
                
            # Calculate basic image statistics
            mean_intensity = np.mean(image)
            std_intensity = np.std(image)
            
            # Simple heuristic-based prediction
            config = self.image_configs.get(image_type, self.image_configs['chest_xray'])
            classes = config['classes']
            
            # Use image statistics for basic classification
            if mean_intensity > 128:  # Brighter images
                predicted_class = classes[0]  # Normal
                confidence = 0.6
            else:  # Darker images
                predicted_class = classes[1]  # Abnormal
                confidence = 0.65
                
            analysis = {
                'prediction': predicted_class,
                'confidence': confidence,
                'all_probabilities': {
                    classes[0]: 1 - confidence,
                    classes[1]: confidence
                },
                'image_type': image_type,
                'model_used': 'Heuristic Analysis (Fallback)',
                'note': 'Deep learning model not available. Using basic analysis.'
            }
            
            return analysis
            
        except Exception as e:
            return {
                'prediction': 'Analysis Failed',
                'confidence': 0.0,
                'error': str(e),
                'image_type': image_type,
                'model_used': 'Error'
            }
    
    def load_image_model(self, image_type):
        """Load pre-trained image model"""
        if not TENSORFLOW_AVAILABLE:
            return False
            
        model_path = f"models/deep_learning/{image_type}_image_model.h5"
        scaler_path = f"models/deep_learning/{image_type}_scaler.pkl"
        
        try:
            if os.path.exists(model_path):
                model = keras.models.load_model(model_path)
                self.models[image_type] = model
                
                if os.path.exists(scaler_path):
                    scaler = pickle.load(open(scaler_path, 'rb'))
                    self.scalers[image_type] = scaler
                
                print(f"✅ Loaded {image_type} image model")
                return True
            else:
                print(f"⚠️ Model file not found: {model_path}")
                return False
                
        except Exception as e:
            print(f"❌ Error loading {image_type} model: {e}")
            return False
    
    def load_model(self, image_type):
        """Backward compatibility method - delegates to load_image_model"""
        return self.load_image_model(image_type)
    
    def explain_image_prediction(self, image_path, image_type='chest_xray', prediction_result=None):
        """
        Provide explainable AI analysis for medical image predictions
        """
        if prediction_result is None:
            prediction_result = self.analyze_medical_image(image_path, image_type)
        
        # Medical knowledge bases for different image types
        medical_explanations = {
            'chest_xray': {
                'Normal': {
                    'description': 'Your chest X-ray shows normal lung and heart structures.',
                    'risk_factors': ['No immediate respiratory concerns detected'],
                    'recommendations': [
                        'Maintain regular exercise',
                        'Avoid smoking and secondhand smoke',
                        'Consider annual chest screenings if over 40',
                        'Practice good respiratory hygiene'
                    ],
                    'precautions': [
                        'Continue healthy lifestyle habits',
                        'Monitor for respiratory symptoms',
                        'Consult doctor if breathing difficulties arise'
                    ]
                },
                'Pneumonia': {
                    'description': 'Your chest X-ray shows patterns consistent with pneumonia.',
                    'risk_factors': [
                        'Consolidation or opacity in lung fields',
                        'Possible inflammation markers',
                        'Altered lung texture patterns'
                    ],
                    'recommendations': [
                        '🚨 CRITICAL: Seek immediate medical attention',
                        'Antibiotic treatment may be required',
                        'Rest and adequate fluid intake',
                        'Monitor oxygen levels if available'
                    ],
                    'precautions': [
                        '⚠️ Do not delay medical consultation',
                        'Avoid strenuous activities',
                        'Practice infection control measures',
                        'Consider hospitalization if severe'
                    ]
                }
            },
            'brain_mri': {
                'Normal': {
                    'description': 'Your brain MRI shows normal brain tissue and structures.',
                    'risk_factors': ['No apparent abnormalities detected'],
                    'recommendations': [
                        'Maintain brain health through regular exercise',
                        'Follow a Mediterranean-style diet',
                        'Engage in cognitive activities',
                        'Ensure adequate sleep (7-9 hours)'
                    ],
                    'precautions': [
                        'Monitor for neurological symptoms',
                        'Maintain cardiovascular health',
                        'Consider regular check-ups if family history exists'
                    ]
                },
                'Tumor': {
                    'description': 'Your brain MRI shows abnormal tissue that may indicate a tumor.',
                    'risk_factors': [
                        'Abnormal tissue growth detected',
                        'Altered brain structure patterns',
                        'Possible mass effect indicators'
                    ],
                    'recommendations': [
                        '🚨 URGENT: Immediate neurological consultation required',
                        'Additional imaging studies may be needed',
                        'Prepare for potential biopsy or surgery',
                        'Gather complete medical history'
                    ],
                    'precautions': [
                        '⚠️ This requires immediate medical attention',
                        'Avoid driving if experiencing symptoms',
                        'Have someone accompany you to appointments',
                        'Monitor for seizures or severe headaches'
                    ]
                }
            },
            'skin_lesion': {
                'Benign': {
                    'description': 'Your skin lesion appears to be benign (non-cancerous).',
                    'risk_factors': ['Low malignancy risk patterns detected'],
                    'recommendations': [
                        'Continue regular self-examinations',
                        'Annual dermatological check-ups',
                        'Use sunscreen daily (SPF 30+)',
                        'Monitor for any changes in size, color, or shape'
                    ],
                    'precautions': [
                        'Watch for ABCDE changes in moles',
                        'Protect skin from UV radiation',
                        'Consult dermatologist if changes occur'
                    ]
                },
                'Malignant': {
                    'description': 'Your skin lesion shows concerning features that may indicate malignancy.',
                    'risk_factors': [
                        'Irregular borders or asymmetry',
                        'Color variations within lesion',
                        'Rapid growth or texture changes'
                    ],
                    'recommendations': [
                        '🚨 URGENT: Dermatological consultation required',
                        'Biopsy may be necessary for confirmation',
                        'Early treatment significantly improves outcomes',
                        'Avoid sun exposure to the area'
                    ],
                    'precautions': [
                        '⚠️ Do not delay professional evaluation',
                        'Protect lesion from trauma',
                        'Document changes with photos',
                        'Consider immediate dermatology referral'
                    ]
                }
            },
            'retinal_scan': {
                'Normal': {
                    'description': 'Your retinal scan shows healthy blood vessels and optic structures.',
                    'risk_factors': ['No diabetic retinopathy signs detected'],
                    'recommendations': [
                        'Maintain good diabetes control if diabetic',
                        'Regular eye examinations (annually)',
                        'Control blood pressure and cholesterol',
                        'Protect eyes from UV damage'
                    ],
                    'precautions': [
                        'Monitor vision changes',
                        'Follow diabetes management plan',
                        'Consider dilated eye exams annually'
                    ]
                },
                'Diabetic Retinopathy': {
                    'description': 'Your retinal scan shows signs of diabetic retinopathy.',
                    'risk_factors': [
                        'Blood vessel damage from diabetes',
                        'Microaneurysms or hemorrhages present',
                        'Possible macular involvement'
                    ],
                    'recommendations': [
                        '🚨 URGENT: Ophthalmological consultation needed',
                        'Optimize diabetes control immediately',
                        'Monitor blood pressure closely',
                        'Consider laser treatment or injections'
                    ],
                    'precautions': [
                        '⚠️ Risk of vision loss without treatment',
                        'Strict blood sugar management essential',
                        'Regular retinal monitoring required',
                        'Avoid activities that increase eye pressure'
                    ]
                }
            }
        }
        
        # Get explanation based on image type and prediction
        explanation = medical_explanations.get(image_type, {}).get(
            prediction_result['prediction'], 
            {
                'description': 'Analysis completed. Please consult healthcare provider.',
                'risk_factors': ['Consultation with medical professional recommended'],
                'recommendations': ['Seek professional medical advice'],
                'precautions': ['Follow standard medical guidelines']
            }
        )
        
        # Add confidence-based risk assessment
        confidence = prediction_result.get('confidence', 0.5)
        risk_level = 'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low'
        
        return {
            'image_type': image_type,
            'prediction': prediction_result['prediction'],
            'confidence': confidence,
            'risk_level': risk_level,
            'explanation': explanation,
            'technical_details': {
                'model_used': prediction_result.get('model_used', 'Unknown'),
                'all_probabilities': prediction_result.get('all_probabilities', {}),
                'analysis_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
    
    def generate_image_report(self, analysis_result):
        """Generate a comprehensive medical imaging report"""
        prediction = analysis_result['prediction']
        confidence = analysis_result['confidence']
        risk_level = analysis_result['risk_level']
        explanation = analysis_result['explanation']
        
        # Risk level colors and emojis
        risk_colors = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
        risk_emoji = risk_colors.get(risk_level, '🔵')
        
        report = f"""
        ## 🏥 Medical Image Analysis Report
        
        **Image Type:** {analysis_result['image_type'].replace('_', ' ').title()}
        **Prediction:** {prediction}
        **Confidence Level:** {confidence:.1%}
        **Risk Assessment:** {risk_emoji} {risk_level} Risk
        
        ### 📋 Clinical Analysis
        {explanation['description']}
        
        ### ⚠️ Key Risk Factors Identified:
        """
        
        for factor in explanation['risk_factors']:
            report += f"• {factor}\n"
        
        report += "\n### 💊 Medical Recommendations:\n"
        for rec in explanation['recommendations']:
            if '🚨' in rec or 'CRITICAL' in rec or 'URGENT' in rec:
                report += f"**{rec}**\n"
            else:
                report += f"• {rec}\n"
        
        report += "\n### 🛡️ Precautionary Measures:\n"
        for precaution in explanation['precautions']:
            if '⚠️' in precaution:
                report += f"**{precaution}**\n"
            else:
                report += f"• {precaution}\n"
        
        report += f"""
        
        ### 🔬 Technical Details
        • Model Used: {analysis_result['technical_details']['model_used']}
        • Analysis Time: {analysis_result['technical_details']['analysis_timestamp']}
        • Confidence Score: {confidence:.3f}
        
        **Medical Disclaimer:** This AI analysis is for screening purposes only. 
        Please consult with qualified healthcare professionals for proper diagnosis and treatment.
        """
        
        return report
    
    def get_model_info(self):
        """Get information about loaded models"""
        info = {
            'tensorflow_available': TENSORFLOW_AVAILABLE,
            'loaded_models': list(self.models.keys()),
            'available_image_types': list(self.image_configs.keys()),
        }
        
        if TENSORFLOW_AVAILABLE:
            info['tensorflow_version'] = tf.__version__
            
        return info
    
    def train_image_model(self, image_type, X, y, epochs=50):
        """Train medical image analysis model"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for training image models")
            
        from sklearn.model_selection import train_test_split
        import json
        
        print(f"Training {image_type} model...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create model
        config = self.image_configs.get(image_type, self.image_configs['chest_xray'])
        model = self.create_cnn_model(config['input_shape'])
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=[
                callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]
        )
        
        # Evaluate
        test_pred = (model.predict(X_test) > 0.5).astype(int)
        accuracy = accuracy_score(y_test, test_pred)
        
        # Save model
        model_path = f"models/deep_learning/{image_type}_image_model.h5"
        model.save(model_path)
        
        # Store in class
        self.models[image_type] = model
        
        print(f"✅ {image_type} model trained with accuracy: {accuracy:.4f}")
        
        return model, accuracy, history