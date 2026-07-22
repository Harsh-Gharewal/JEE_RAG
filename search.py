from __future__ import annotations

import logging
from typing import List, Dict
from urllib.parse import urlparse

from tavily import TavilyClient

from config import get_settings


logger = logging.getLogger("jee_rag")


def search_web(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    settings = get_settings()
    client = TavilyClient(api_key=settings.tavily_api_key)
    response = client.search(
        query=query,
        max_results=max_results,
        include_answer=False,
        include_raw_content=True,
    )
    blocked_domains = {"facebook.com", "chegg.com"}
    results = []
    for item in response.get("results", []):
        url = item.get("url", "")
        hostname = urlparse(url).hostname or ""
        title = item.get("title", "")
        snippet = item.get("snippet", "") or ""
        content = item.get("content", "") or ""
        if not content.strip():
            continue
        if not snippet:
            snippet = content[:200].replace("\n", " ").strip()
        results.append(
            {
                "url": url,
                "title": title,
                "snippet": snippet,
                "content": content,
            }
        )
    if results:
        filtered = [
            item
            for item in results
            if not any(
                (urlparse(item["url"]).hostname or "").endswith(domain)
                for domain in blocked_domains
            )
        ]
        if filtered:
            results = filtered
    logger.info("Tavily results: %s, usable: %s", len(response.get("results", [])), len(results))
    return results
