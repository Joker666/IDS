import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from utils import load_data

def preprocess_data(df, is_training=True, existing_encoders=None, existing_scaler=None):
    """Preprocess the data for machine learning"""
    # Create copy to avoid modifying original data
    df_ml = df.copy()

    # Convert categorical variables to numerical
    categorical_columns = ['protocol_type', 'service', 'flag']
    if is_training:
        categorical_columns.append('class')

    label_encoders = existing_encoders if existing_encoders else {}

    for column in categorical_columns:
        if column in df_ml.columns:  # Only process columns that exist in the dataset
            if is_training or column not in label_encoders:
                label_encoders[column] = LabelEncoder()
                # Fit and transform
                df_ml[column] = label_encoders[column].fit_transform(df_ml[column])
            else:
                # Handle unseen categories for test data
                known_categories = label_encoders[column].classes_
                df_ml[column] = df_ml[column].map(lambda x: x if x in known_categories else known_categories[0])
                df_ml[column] = label_encoders[column].transform(df_ml[column])

    # Split features and target if in training mode
    if is_training:
        X = df_ml.drop('class', axis=1)
        y = df_ml['class']
    else:
        X = df_ml
        y = None

    # Scale numerical features
    scaler = existing_scaler if existing_scaler else StandardScaler()
    X_scaled = scaler.fit_transform(X) if is_training else scaler.transform(X)

    return X_scaled, y, label_encoders, scaler

def train_model(X, y):
    """Train a Random Forest model using training data"""
    # Train model on full training data
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Make predictions on training data for initial evaluation
    y_pred = model.predict(X)

    # Get evaluation metrics
    report = classification_report(y, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y, y_pred)

    return model, report, conf_matrix

def save_model(model, label_encoders, scaler, filename='model.joblib'):
    """Save the trained model and preprocessing objects"""
    model_data = {
        'model': model,
        'label_encoders': label_encoders,
        'scaler': scaler
    }
    joblib.dump(model_data, filename)

def load_model(filename='model.joblib'):
    """Load the trained model and preprocessing objects"""
    model_data = joblib.load(filename)
    return model_data['model'], model_data['label_encoders'], model_data['scaler']

def make_prediction(data, model, label_encoders, scaler):
    """Make predictions on new data"""
    # Preprocess the input data
    data_processed = data.copy()

    # Encode categorical variables
    for column, encoder in label_encoders.items():
        if column != 'class' and column in data_processed.columns:
            # Handle unseen categories
            known_categories = encoder.classes_
            data_processed[column] = data_processed[column].map(lambda x: x if x in known_categories else known_categories[0])
            data_processed[column] = encoder.transform(data_processed[column])

    # Scale features
    data_scaled = scaler.transform(data_processed)

    # Make prediction
    prediction = model.predict(data_scaled)
    probabilities = model.predict_proba(data_scaled)

    return prediction, probabilities