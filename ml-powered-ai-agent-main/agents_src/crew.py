from crewai import Crew

from agents_src.agents.diabetes_prediction_agent import diabetes_prediction_agent
from agents_src.tasks.diabetes_prediction_task import predict_diabetes_task

diabetes_prediction_crew = Crew(
    agents=[diabetes_prediction_agent],
    tasks=[predict_diabetes_task],
    verbose=True
)
