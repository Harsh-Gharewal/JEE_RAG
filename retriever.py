from __future__ import annotations

import asyncio
import math
from collections import Counter
from typing import Dict, List, Tuple
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from config import get_settings
from embedder import embed_documents, embed_query


def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
    max_chars: int = 6000,
    max_chunks: int = 8,
) -> List[str]:
    try:
        trimmed = text[:max_chars]
        words = trimmed.split()
        if not words:
            return []
        chunks = []
        start = 0
        while start < len(words) and len(chunks) < max_chunks:
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            if chunk:
                chunks.append(chunk)
            start = end - overlap
            if start < 0:
                start = 0
            if start >= len(words):
                break
        return chunks
    except MemoryError:
        return []


def _build_points(
    chunks: List[Dict[str, str]],
    embeddings: List[List[float]],
) -> List[PointStruct]:
    points = []
    for chunk, vector in zip(chunks, embeddings):
        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload={
                    "url": chunk["url"],
                    "title": chunk["title"],
                    "chunk_text": chunk["chunk_text"],
                    "request_id": chunk["request_id"],
                },
            )
        )
    return points


def _tokenize(text: str) -> List[str]:
    return [token for token in text.lower().split() if token]


def _bm25_scores(query: str, documents: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    if not documents:
        return []
    doc_tokens = [_tokenize(doc) for doc in documents]
    doc_lens = [len(tokens) for tokens in doc_tokens]
    avgdl = sum(doc_lens) / max(len(doc_lens), 1)

    df = Counter()
    for tokens in doc_tokens:
        for term in set(tokens):
            df[term] += 1

    query_terms = _tokenize(query)
    scores = []
    num_docs = len(documents)
    for tokens, dl in zip(doc_tokens, doc_lens):
        tf = Counter(tokens)
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            term_df = df.get(term, 0)
            idf = math.log((num_docs - term_df + 0.5) / (term_df + 0.5) + 1.0)
            denom = tf[term] + k1 * (1 - b + b * (dl / max(avgdl, 1.0)))
            score += idf * (tf[term] * (k1 + 1)) / max(denom, 1e-8)
        scores.append(score)
    return scores


async def retrieve_chunks(
    question: str,
    search_results: List[Dict[str, str]],
    request_id: str,
    top_k: int = 5,
) -> Tuple[str, List[Dict[str, str]]]:
    settings = get_settings()
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )

    collection_name = settings.qdrant_collection
    collections = await asyncio.to_thread(client.get_collections)
    existing = {collection.name for collection in collections.collections}
    if collection_name not in existing:
        await asyncio.to_thread(
            client.create_collection,
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        try:
            await asyncio.to_thread(
                client.create_payload_index,
                collection_name=collection_name,
                field_name="request_id",
                field_schema="keyword",
            )
        except Exception:
            pass

    chunk_records: List[Dict[str, str]] = []
    seen_chunks = set()
    max_total_chunks = 10
    for result in search_results:
        if len(chunk_records) >= max_total_chunks:
            break
        for chunk in chunk_text(result["content"]):
            key = (result["url"], chunk)
            if key in seen_chunks:
                continue
            seen_chunks.add(key)
            chunk_records.append(
                {
                    "url": result["url"],
                    "title": result["title"],
                    "chunk_text": chunk,
                    "request_id": request_id,
                }
            )
            if len(chunk_records) >= max_total_chunks:
                break

    if not chunk_records:
        return collection_name, []

    embeddings = await embed_documents([c["chunk_text"] for c in chunk_records])
    points = _build_points(chunk_records, embeddings)
    await asyncio.to_thread(client.upsert, collection_name=collection_name, points=points)

    query_vector = await embed_query(question)
    query_filter = Filter(
        must=[FieldCondition(key="request_id", match=MatchValue(value=request_id))]
    )
    hits = await asyncio.to_thread(
        client.query_points,
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        query_filter=query_filter,
    )
    if hasattr(hits, "points"):
        hits = hits.points
    elif hasattr(hits, "result"):
        hits = hits.result

    dense_hits = []
    for hit in hits:
        payload = getattr(hit, "payload", None) or {}
        score = getattr(hit, "score", None)
        dense_hits.append(
            {
                "url": payload.get("url", ""),
                "title": payload.get("title", ""),
                "chunk_text": payload.get("chunk_text", ""),
                "score": score or 0.0,
            }
        )

    bm25_scores = _bm25_scores(question, [c["chunk_text"] for c in chunk_records])
    bm25_ranked = sorted(
        zip(chunk_records, bm25_scores), key=lambda item: item[1], reverse=True
    )
    bm25_top = bm25_ranked[:top_k]

    candidates: Dict[Tuple[str, str], Dict[str, object]] = {}
    for item in dense_hits:
        key = (item["url"], item["chunk_text"])
        candidates[key] = {
            "url": item["url"],
            "title": item["title"],
            "chunk_text": item["chunk_text"],
            "dense": float(item["score"]),
            "bm25": 0.0,
        }
    for chunk, score in bm25_top:
        key = (chunk["url"], chunk["chunk_text"])
        entry = candidates.get(key)
        if not entry:
            entry = {
                "url": chunk["url"],
                "title": chunk["title"],
                "chunk_text": chunk["chunk_text"],
                "dense": 0.0,
                "bm25": 0.0,
            }
            candidates[key] = entry
        entry["bm25"] = float(score)

    if not candidates:
        return collection_name, []

    dense_vals = [item["dense"] for item in candidates.values()]
    bm25_vals = [item["bm25"] for item in candidates.values()]
    dense_min, dense_max = min(dense_vals), max(dense_vals)
    bm25_min, bm25_max = min(bm25_vals), max(bm25_vals)

    def normalize(value: float, vmin: float, vmax: float) -> float:
        if vmax <= vmin:
            return 0.0
        return (value - vmin) / (vmax - vmin)

    results = []
    for item in candidates.values():
        dense_norm = normalize(item["dense"], dense_min, dense_max)
        bm25_norm = normalize(item["bm25"], bm25_min, bm25_max)
        combined = 0.6 * dense_norm + 0.4 * bm25_norm
        results.append(
            {
                "url": item["url"],
                "title": item["title"],
                "chunk_text": item["chunk_text"],
                "score": combined,
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return collection_name, results[:top_k]
