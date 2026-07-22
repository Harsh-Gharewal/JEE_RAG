from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, examples=["A particle moves in a circle of radius R. What is the centripetal acceleration if speed is v?"]
    )


class Source(BaseModel):
    url: str
    title: str
    snippet: str


class ContextChunk(BaseModel):
    url: str
    title: str
    chunk_text: str
    score: float | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source] = Field(default_factory=list)
    context_chunks: List[ContextChunk] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "question": "A particle moves in a circle of radius R. What is the centripetal acceleration if speed is v?",
                "answer": "Using circular motion, centripetal acceleration a = v^2 / R.",
                "sources": [
                    {
                        "url": "https://example.com/circular-motion",
                        "title": "Uniform Circular Motion",
                        "snippet": "Centripetal acceleration is v^2/R for constant speed.",
                    }
                ],
                "context_chunks": [
                    {
                        "url": "https://example.com/circular-motion",
                        "title": "Uniform Circular Motion",
                        "chunk_text": "For a body in uniform circular motion, the centripetal acceleration is v^2/R.",
                        "score": 0.92,
                    }
                ],
            }
        }
