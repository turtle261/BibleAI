import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl

class Settings(BaseSettings):
    # API Settings
    APP_NAME: str = "Bible AI - Semantic Search & LLM Assistant"
    VERSION: str = "6.0"
    DESCRIPTION: str = "An intelligent Bible study assistant combining semantic search with Ollama LLM capabilities"

    # CORS Settings
    ALLOWED_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost",
        "http://127.0.0.1:8000",
        "http://localhost:8000"
    ]

    # Paths
    BIBLE_DATA_PATH: str = os.path.join("data", "bible.json")
    GREEK_DATA_PATH: str = os.path.join("data", "el_greek.json")
    FAISS_INDEX_PATH: str = os.path.join("data", "embeddings", "faiss.index")

    # Ollama Settings
    MODEL_NAME: str = "llama3.2:3b-instruct-q8_0"  # Tool-capable LLM
    EMBEDDING_MODEL: str = "all-minilm"  # Embedding model
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TIMEOUT: int = 60  # in seconds

    # Semantic Search Settings
    EMBEDDING_BATCH_SIZE: int = 384
    TOP_K_RESULTS: int = 5
    EMBEDDING_DIMENSION: int = 384

    # Logging
    LOG_LEVEL: str = "INFO"

    # Application Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
