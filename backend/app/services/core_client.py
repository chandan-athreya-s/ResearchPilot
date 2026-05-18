from __future__ import annotations

import os
from typing import Any, Dict, List

import requests

from app.services.rate_limit_manager import get_rate_limit_manager
from app.services.retrieval_cache import get_retrieval_cache

CORE_API_KEY = os.getenv("CORE_API_KEY", "")
CORE_API_URL = os.getenv("CORE_API_URL", "https://api.core.ac.uk/v3/search/works")


def _debug_response(response: requests.Response, context: str) -> None:
    print(f"[{context}] STATUS: {response.status_code}")
    print(f"[{context}] HEADERS: {dict(response.headers)}")
    body = getattr(response, "text", "") or ""
    print(f"[{context}] BODY: {body[:500]}")


def _normalize_text(text: str) -> str:
    return "" if not text else " ".join(str(text).strip().split())


def _extract_authors(item: Dict[str, Any]) -> List[str]:
    authors = []
    for author in item.get("authors", []):
        if isinstance(author, dict):
            name = author.get("name") or author.get("displayName")
            if name:
                authors.append(name)
        elif isinstance(author, str):
            authors.append(author)
    return authors


def fetch_core_papers(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    """Fetch paper metadata from CORE when API credentials are configured."""
    if not CORE_API_KEY:
        return []

    rate_limit_mgr = get_rate_limit_manager()
    cache = get_retrieval_cache()
    cached = cache.get(query, "core")
    if cached is not None:
        print(f"[CORE] Cache hit for query: {query[:50]}...")
        return cached

    if not rate_limit_mgr.is_enabled("core"):
        remaining = rate_limit_mgr.get_time_until_retry("core")
        print(f"[CORE] Rate limited. Retry in {remaining:.1f}s")
        return []

    params = {
        "q": query,
        "pageSize": max_results,
        "sort": "relevance",
    }
    headers = {"Authorization": CORE_API_KEY}

    try:
        response = requests.get(CORE_API_URL, headers=headers, params=params, timeout=12)
    except requests.exceptions.RequestException as exc:
        print(f"[CORE] network failure: {exc}")
        rate_limit_mgr.record_network_error("core")
        return []

    if response.status_code == 429:
        retry_after = None
        if "retry-after" in response.headers:
            try:
                retry_after = int(response.headers["retry-after"])
            except ValueError:
                pass
        rate_limit_mgr.record_rate_limit("core", retry_after)
        return []

    if response.status_code != 200:
        _debug_response(response, "CORE non-200 response")
        rate_limit_mgr.record_network_error("core")
        return []

    text = getattr(response, "text", "") or ""
    if not text.strip():
        print("[CORE] empty response body")
        return []

    if "<html" in text.lower() or "<!doctype html" in text.lower():
        _debug_response(response, "CORE HTML error response")
        return []

    try:
        data = response.json()
    except Exception as exc:
        print("CORE JSON PARSE FAILED")
        _debug_response(response, "CORE malformed JSON")
        print(f"[CORE] parse exception: {exc}")
        rate_limit_mgr.record_network_error("core")
        return []

    if not isinstance(data, dict):
        print("[CORE] unexpected response format, expected JSON object")
        _debug_response(response, "CORE unexpected JSON format")
        rate_limit_mgr.record_network_error("core")
        return []

    papers: List[Dict[str, Any]] = []
    for item in data.get("data", []):
        title = _normalize_text(item.get("title", ""))
        if not title:
            continue

        doi = _normalize_text(item.get("doi", ""))
        pdf_url = _normalize_text(item.get("fullTextUrl", "")) or _normalize_text(item.get("url", ""))
        paper_id = _normalize_text(item.get("id", doi or title))

        papers.append({
            "paper_id": paper_id,
            "title": title,
            "abstract": _normalize_text(item.get("abstract", "")),
            "url": _normalize_text(item.get("url", "")),
            "doi": doi,
            "pdf_url": pdf_url,
            "authors": _extract_authors(item),
            "year": item.get("year"),
            "venue": _normalize_text(item.get("publisher", "")),
            "locations": [],
            "open_access": {},
            "source": "core",
        })

    if papers:
        cache.set(query, "core", papers)
        rate_limit_mgr.record_success("core")
        print(f"[CORE] Successfully fetched {len(papers)} papers")

    return papers
