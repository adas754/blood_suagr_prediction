from crewai import Agent, LLM

from agents_src.tools.diabetes_prediction_tool import predict_diabetes
from config.settings import Settings

settings = Settings()
model = settings.DEFAULT_LLM
temperature = float(settings.DEFAULT_TEMPERATURE)

# initialize the llm
llm = LLM(
    model=model,
    temperature=temperature
)

diabetes_prediction_agent = Agent(
    role="Diabetes Prediction Specialist",
    goal=(
        "Accurately predict diabetes risk for patients using only the provided prediction tool,"
        " strictly based on input data."
        " Additionally, provide clear and actionable health advice tailored to the prediction result."
    ),
    backstory=(
        "You are an analytical yet supportive assistant whose purpose is to deliver precise diabetes"
        " risk predictions and practical health advice. "
        "Your persona is objective and caring: you use the prediction tool for risk assessment and offer clear,"
        " actionable recommendations based on the results. "
        "Your output is reliable, concise, and designed to help patients and healthcare professionals take"
        " informed steps toward better health."
    ),
    llm=llm,
    tools=[predict_diabetes],
    verbose=True
)
