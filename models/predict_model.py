import joblib
import pandas as pd
from utils.feature_engineering import create_features
import os

# MODEL_PATH = "models/best_model.pkl"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pkl")

# Load model once
_model = joblib.load(MODEL_PATH)
_model_features = _model.feature_name_  # store all features used in training

def load_model():
    return _model

def predict(input_data: pd.DataFrame):
    """Predict diabetes probability given raw input data."""
    df = create_features(input_data)

    # Ensure all required features exist
    for f in _model_features:
        if f not in df.columns:
            df[f] = 0  # or np.nan if you handle missing values in model

    # Reorder columns exactly like training
    df = df[_model_features]

    model = load_model()
    prob = model.predict_proba(df)[:, 1]
    return prob
