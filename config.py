from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    tavily_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    ollama_base_url: str


def get_settings() -> Settings:
    settings = Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        qdrant_url=os.getenv("QDRANT_URL", ""),
        qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
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
