from __future__ import annotations

import asyncio
from typing import List

import httpx

from config import get_settings


async def _embed_single(client: httpx.AsyncClient, text: str, model: str) -> List[float]:
    response = await client.post(
        "/api/embeddings",
        json={
            "model": model,
            "prompt": text,
        },
    )
    response.raise_for_status()
    data = response.json()
    embedding = data.get("embedding")
    if not embedding:
        raise ValueError("Unexpected embedding response format")
    return embedding


async def _embed_batch(texts: List[str]) -> List[List[float]]:
    settings = get_settings()
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
    timeout = httpx.Timeout(60.0)
    async with httpx.AsyncClient(
        base_url=settings.ollama_base_url, limits=limits, timeout=timeout
    ) as client:
        semaphore = asyncio.Semaphore(8)

        async def guarded_embed(text: str) -> List[float]:
            async with semaphore:
                return await _embed_single(client, text, settings.ollama_embed_model)

        tasks = [guarded_embed(text) for text in texts]
        return await asyncio.gather(*tasks)


async def embed_documents(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    if not texts:
        return []
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    tasks = [_embed_batch(batch) for batch in batches]
    results = await asyncio.gather(*tasks)
    embeddings: List[List[float]] = []
    for batch_embeddings in results:
        embeddings.extend(batch_embeddings)
    return embeddings


async def embed_query(text: str) -> List[float]:
    embeddings = await _embed_batch([text])
    return embeddings[0]
