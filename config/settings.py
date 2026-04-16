from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load env variables
load_dotenv()

class Settings(BaseSettings):
    # Required
    GROQ_API_KEY: str

    # Optional defaults
    DEFAULT_LLM: str = "groq/llama-3.3-70b-versatile"
    DEFAULT_TEMPERATURE: float = 0.0
    DATASET_PATH: str = r"E:\ml-powered-ai-agent-main\artifacts\data\diabetes.csv"
    MODEL_PATH: str = r"E:\ml-powered-ai-agent-main\artifacts\model\diabetes_model.pkl"
    OTEL_SDK_DISABLED: bool = True

    # ✅ Fix for your latest error
    CREWAI_TRACING_ENABLED: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"   # ✅ prevents future crashes