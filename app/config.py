import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    MODEL: str = "llama-3.3-70b-versatile"
    APP_VERSION: str = "0.1.0"

    # document processing
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MAX_SEARCH_RESULTS: int = 10
    MIN_SIMILARITY_THRESHOLD: float = 0.15

    # RAG settings
    DEFAULT_N_CANDIDATES: int = 10
    DEFAULT_FINAL_RESULTS: int = 3
    DEFAULT_ALPHA: float = 0.5
    MODEL: str = os.getenv("DEFAULT_MODEL", "llama-3.3-70b-versatile")

    # LangSmith
    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "ai-engineer-journey")

settings = Settings()