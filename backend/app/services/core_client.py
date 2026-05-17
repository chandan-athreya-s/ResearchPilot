from __future__ import annotations

import os
from typing import Any, Dict, List

import requests

CORE_API_KEY = os.getenv("CORE_API_KEY", "")
CORE_API_URL = os.getenv("CORE_API_URL", "https://api.core.ac.uk/v3/search/works")


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

    params = {
        "q": query,
        "pageSize": max_results,
        "sort": "relevance",
    }
    headers = {"Authorization": CORE_API_KEY}

    try:
        response = requests.get(CORE_API_URL, headers=headers, params=params, timeout=12)
        response.raise_for_status()
        data = response.json()
    except Exception:
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

    return papers
