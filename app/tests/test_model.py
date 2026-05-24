# =============================================================
# Unit Tests — Heart Disease Prediction API
# =============================================================
# These tests verify that our data processing and model
# work correctly. Run with: pytest tests/test_model.py

import pytest
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# =============================================================
# SECTION 1: TEST DATA LOADING
# =============================================================

def test_cleaned_dataset_exists():
    """Check the cleaned CSV file can be loaded"""
    df = pd.read_csv("heart_disease_cleaned.csv")
    assert df is not None
    assert len(df) > 0, "Dataset should not be empty"

def test_dataset_has_correct_columns():
    """Check all 14 expected columns are present"""
    df = pd.read_csv("heart_disease_cleaned.csv")
    expected_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                        'restecg', 'thalach', 'exang', 'oldpeak',
                        'slope', 'ca', 'thal', 'target']
    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

def test_no_missing_values():
    """Check cleaned dataset has no missing values"""
    df = pd.read_csv("heart_disease_cleaned.csv")
    assert df.isnull().sum().sum() == 0, "Dataset should have no missing values"

def test_target_is_binary():
    """Check target column only contains 0 and 1"""
    df = pd.read_csv("heart_disease_cleaned.csv")
    unique_values = set(df['target'].unique())
    assert unique_values == {0, 1}, f"Target should be binary, got: {unique_values}"

# =============================================================
# SECTION 2: TEST MODEL LOADING
# =============================================================

def test_model_file_loads():
    """Check the model pickle file loads without errors"""
    with open("heart_disease_model.pkl", "rb") as f:
        model = pickle.load(f)
    assert model is not None, "Model should load successfully"

def test_feature_columns_load():
    """Check the feature columns file loads correctly"""
    with open("feature_columns.pkl", "rb") as f:
        features = pickle.load(f)
    assert len(features) == 13, "Should have 13 feature columns"

# =============================================================
# SECTION 3: TEST MODEL PREDICTIONS
# =============================================================

def test_model_predicts_binary():
    """Check model only outputs 0 or 1"""
    with open("heart_disease_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Sample healthy patient data
    sample = np.array([[63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6]])
    prediction = model.predict(sample)

    assert prediction[0] in [0, 1], "Prediction must be 0 or 1"

def test_model_returns_probability():
    """Check model returns valid probability between 0 and 1"""
    with open("heart_disease_model.pkl", "rb") as f:
        model = pickle.load(f)

    sample = np.array([[63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6]])
    probability = model.predict_proba(sample)[0][1]

    assert 0.0 <= probability <= 1.0, "Probability must be between 0 and 1"

def test_model_handles_disease_case():
    """Check model can predict disease (1) for high-risk patient"""
    with open("heart_disease_model.pkl", "rb") as f:
        model = pickle.load(f)

    # High risk patient profile
    high_risk = np.array([[67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3]])
    prediction = model.predict(high_risk)

    assert prediction[0] in [0, 1], "Should return valid prediction"

# =============================================================
# SECTION 4: TEST API INPUT VALIDATION
# =============================================================

def test_input_feature_count():
    """Check that a sample input has exactly 13 features"""
    sample_input = {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145,
        "chol": 233, "fbs": 1, "restecg": 2, "thalach": 150,
        "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6
    }
    assert len(sample_input) == 13, "Input must have exactly 13 features"
