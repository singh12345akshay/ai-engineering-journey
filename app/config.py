import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    MODEL: str = "llama-3.3-70b-versatile"
    APP_VERSION: str = "0.1.0"

settings = Settings()