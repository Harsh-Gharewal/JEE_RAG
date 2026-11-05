from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    tavily_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"
    groq_model: str = "llama-3.3-70b-versatile"
    qdrant_collection: str = "jee-rag"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        qdrant_url=os.getenv("QDRANT_URL", ""),
        qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "jee-rag"),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
    missing = [
        name
        for name, value in {
            "GROQ_API_KEY": settings.groq_api_key,
            "TAVILY_API_KEY": settings.tavily_api_key,
            "QDRANT_URL": settings.qdrant_url,
            "QDRANT_API_KEY": settings.qdrant_api_key,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    return settings
