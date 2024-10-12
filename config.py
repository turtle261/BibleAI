import os
from typing import List

from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl
class Settings(BaseSettings):
    # API Settings
    APP_NAME: str = "Optimized FOSS Bible AI API"
    VERSION: str = "5.0"
    DESCRIPTION: str = "An advanced, memory-optimized AI-powered Bible study and theological research tool using Ollama as the inference engine."

    # CORS Settings
    ALLOWED_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost",
        "http://127.0.0.1:8000"
    ]

    # Paths
    BIBLE_DATA_PATH: str = "data/bible.json"
    FAISS_INDEX_PATH: str = "data/embeddings/faiss.index"

    # Ollama Settings
    MODEL_NAME: str = "llama3.2"
    OLLAMA_TIMEOUT: int = 60  # in seconds

    # Logging
    LOG_LEVEL: str = "WARNING"

    # Application Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    class Config:
        env_file = ".env"

settings = Settings()