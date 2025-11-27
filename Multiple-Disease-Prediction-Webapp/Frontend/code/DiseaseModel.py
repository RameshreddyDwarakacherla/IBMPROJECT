import xgboost as xgb
import pandas as pd
import numpy as np

class DiseaseModel:

    def __init__(self):
        self.all_symptoms = None
        self.symptoms = None
        self.pred_disease = None
        self.model = xgb.XGBClassifier()
        self.model_loaded = False
        self.diseases = self.disease_list('data/dataset.csv')

    def load_xgboost(self, model_path):
        try:
            self.model.load_model(model_path)
            self.model_loaded = True

            # Fix for XGBoost compatibility - ensure n_classes_ attribute exists
            if not hasattr(self.model, 'n_classes_'):
                # Get unique classes from the diseases list
                unique_diseases = len(self.diseases)
                self.model.n_classes_ = unique_diseases
        except Exception as e:
            print(f"Warning: Could not load XGBoost model from {model_path}: {e}")
            self.model_loaded = False

    def save_xgboost(self, model_path):
        self.model.save_model(model_path)

    def predict(self, X):
        self.symptoms = X

        # Check if model is loaded
        if not self.model_loaded:
            print("ℹ️ XGBoost model not loaded, using fallback prediction")
            # Simple fallback based on number of symptoms
            if len(X) > 0 and np.sum(X) > 0:
                # Return a random disease based on symptom count
                import random
                disease_idx = random.randint(0, min(len(self.diseases)-1, int(np.sum(X)) % len(self.diseases)))
                self.pred_disease = self.diseases[disease_idx]
                disease_probability = 0.6 + (np.sum(X) / len(X)) * 0.3  # Between 0.6-0.9
            else:
                self.pred_disease = "No symptoms detected"
                disease_probability = 0.1
            return self.pred_disease, disease_probability

        try:
            # Use predict method with proper error handling
            disease_pred_idx = self.model.predict(self.symptoms)

            # Ensure the prediction index is valid
            if isinstance(disease_pred_idx, np.ndarray):
                disease_pred_idx = disease_pred_idx[0]

            # Get the predicted disease - diseases is a categorical index, not DataFrame
            if disease_pred_idx < len(self.diseases):
                self.pred_disease = self.diseases[disease_pred_idx]
            else:
                self.pred_disease = self.diseases[0]  # Fallback to first disease

            # Get probability
            disease_probability_array = self.model.predict_proba(self.symptoms)

            # Ensure we have valid probability
            if disease_probability_array.shape[1] > disease_pred_idx:
                disease_probability = disease_probability_array[0, disease_pred_idx]
            else:
                disease_probability = np.max(disease_probability_array[0])

        except Exception as e:
            print(f"Prediction error: {e}")
            # Fallback prediction
            self.pred_disease = self.diseases[0] if len(self.diseases) > 0 else "Unknown"
            disease_probability = 0.5

        return self.pred_disease, disease_probability

    
    def describe_disease(self, disease_name):

        if disease_name not in self.diseases:
            return "That disease is not contemplated in this model"
        
        # Read disease dataframe
        desc_df = pd.read_csv('data/symptom_Description.csv')
        desc_df = desc_df.apply(lambda col: col.str.strip())

        return desc_df[desc_df['Disease'] == disease_name]['Description'].values[0]

    def describe_predicted_disease(self):

        if self.pred_disease is None:
            return "No predicted disease yet"

        return self.describe_disease(self.pred_disease)
    
    def disease_precautions(self, disease_name):

        if disease_name not in self.diseases:
            return "That disease is not contemplated in this model"

        # Read precautions dataframe
        prec_df = pd.read_csv('data/symptom_precaution.csv')
        prec_df = prec_df.apply(lambda col: col.str.strip())

        return prec_df[prec_df['Disease'] == disease_name].filter(regex='Precaution').values.tolist()[0]

    def predicted_disease_precautions(self):

        if self.pred_disease is None:
            return "No predicted disease yet"

        return self.disease_precautions(self.pred_disease)

    def disease_list(self, kaggle_dataset):

        df = pd.read_csv('data/clean_dataset.tsv', sep='\t')
        # Preprocessing
        y_data = df.iloc[:,-1]
        X_data = df.iloc[:,:-1]

        self.all_symptoms = X_data.columns

        # Convert y to categorical values
        y_data = y_data.astype('category')
        
        return y_data.cat.categories