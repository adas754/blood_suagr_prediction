from pydantic import BaseModel
from typing import List, Any
from crewai import Task
from agents_src.agents.diabetes_prediction_agent import diabetes_prediction_agent


class DiabetesRiskPrediction(BaseModel):
    result: str  # 'Positive' or 'Negative'
    probability_diabetic: float
    test_result_natural_language: str  # Human-readable summary of the test result

class DiabetesPredictionOutput(BaseModel):
    diabetes_risk_prediction: DiabetesRiskPrediction
    explanation: List[str]
    actionable_health_advice: List[str]
    tool_output: Any


predict_diabetes_task = Task(
    description=(
        "Predict the risk of diabetes for a patient using the following input data: "
        "patient_data: {patient_data}"
        "\nUse the diabetes prediction tool to assess risk strictly based on these inputs. "
        "Provide a clear summary of the prediction and offer actionable health advice tailored to the result."
    ),
    expected_output=(
        "A Python dict with the following structure: "
        "{'diabetes_risk_prediction': {'result': <'Positive' or 'Negative'>, 'probability': <float>, 'test_result_natural_language': <str>}, "
        "'explanation': [<bullet-pointed explanations>], "
        "'actionable_health_advice': [<bullet-pointed recommendations>], "
        "'tool_output': <raw output from the diabetes prediction tool>}"
        "\nThe 'test_result_natural_language' field should provide a human-readable summary of the test result."
    ),
    output_pydantic=DiabetesPredictionOutput,
    agent=diabetes_prediction_agent
)
