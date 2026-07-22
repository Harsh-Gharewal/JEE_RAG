from __future__ import annotations

import logging

from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from generator import generate_answer
from models import AskRequest, AskResponse, ContextChunk, Source
from retriever import retrieve_chunks
from search import search_web

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jee_rag")

app = FastAPI(title="JEE RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskRequest) -> AskResponse:
    try:
        request_id = uuid4().hex
        search_results = search_web(payload.question, max_results=10)
        collection_name, context_chunks = await retrieve_chunks(
            payload.question, search_results, request_id, top_k=5
        )
        _ = collection_name
        answer = generate_answer(payload.question, context_chunks)

        sources = [
            Source(url=item["url"], title=item["title"], snippet=item["snippet"])
            for item in search_results
        ]
        context_models = [
            ContextChunk(
                url=item["url"],
                title=item["title"],
                chunk_text=item["chunk_text"],
                score=item.get("score"),
            )
            for item in context_chunks
        ]

        return AskResponse(
            question=payload.question,
            answer=answer,
            sources=sources,
            context_chunks=context_models,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unhandled error while processing /ask")
        raise HTTPException(status_code=500, detail="Internal server error") from exc
