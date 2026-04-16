from crewai.tools import tool
import joblib
import pandas as pd

from config.settings import Settings

# Load settings and model once at module level
settings = Settings()
model_path = settings.MODEL_PATH
model = joblib.load(model_path)


@tool("Diabetes Prediction Tool")
def predict_diabetes(patient_data: dict) -> dict:
    """
    Predict diabetes outcome for new patient data.

    Args:
        patient_data (dict): Dictionary with patient features:
            {
                "Pregnancies": int,
                "Glucose": float,
                "BloodPressure": float,
                "SkinThickness": float,
                "Insulin": float,
                "BMI": float,
                "DiabetesPedigreeFunction": float,
                "Age": int
            }

    Returns:
        dict: Prediction result with class, label, and probability.
    """
    # Convert dict → DataFrame (1 row)
    df = pd.DataFrame([patient_data])

    # Predict
    pred_class = model.predict(df)[0]
    pred_prob = round(model.predict_proba(df)[0][1], 2)

    # 0 = Non-diabetic, 1 = Diabetic
    # Add human-readable label
    label = "Diabetic" if pred_class == 1 else "Non-Diabetic"

    return {
        "prediction": int(pred_class),
        "label": label,
        "probability_diabetic": float(pred_prob)
    }
