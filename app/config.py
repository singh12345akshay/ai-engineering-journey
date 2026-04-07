import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    MODEL: str = "llama-3.3-70b-versatile"
    APP_VERSION: str = "0.1.0"

    # document processing
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MAX_SEARCH_RESULTS: int = 10
    MIN_SIMILARITY_THRESHOLD: float = 0.15

settings = Settings()