from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from app.agents.relevance_verifier_agent import compute_relevance_score, extract_query_entities
from app.services.core_client import fetch_core_papers
from app.services.openalex_client import fetch_papers as fetch_openalex_papers
from app.services.rate_limit_manager import get_rate_limit_manager

DOMAIN_BOOST_TERMS = [
    "retrieval augmented generation",
    "rag",
    "large language model",
    "llm",
    "retrieval systems",
    "autonomous agent",
    "agentic workflow",
    "multi-agent",
    "enterprise",
    "knowledge system",
    "enterprise ai",
]

LOW_RELEVANCE_PATTERNS = [
    r"\bfederated learning\b",
    r"\bbiomedical\b",
    r"\bxai\b",
    r"\bexplained?able ai\b",
    r"\bmedical\b",
    r"\bhealthcare\b",
    r"\bgeneric ai\b",
]


def _normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def _title_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _paper_key(paper: Dict[str, Any]) -> str:
    doi = _normalize_text(paper.get("doi") or "")
    if doi:
        return doi
    title = _normalize_text(paper.get("title") or paper.get("paper_id") or "")
    return title


def _is_same_paper(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    key_a = _paper_key(a)
    key_b = _paper_key(b)
    if key_a and key_b and key_a == key_b:
        return True
    title_a = _normalize_text(a.get("title"))
    title_b = _normalize_text(b.get("title"))
    if title_a and title_b and _title_similarity(title_a, title_b) > 0.9:
        return True
    return False


def _choose_better_paper(primary: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    score_primary = len(_normalize_text(primary.get("abstract", "")))
    score_candidate = len(_normalize_text(candidate.get("abstract", "")))
    primary_oa = bool(primary.get("pdf_url") or primary.get("open_access"))
    candidate_oa = bool(candidate.get("pdf_url") or candidate.get("open_access"))

    if candidate_oa and not primary_oa:
        return candidate
    if score_candidate > score_primary + 50:
        return candidate
    if candidate.get("year") and primary.get("year") and candidate.get("year") > primary.get("year"):
        return candidate
    return primary


def deduplicate_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique: List[Dict[str, Any]] = []

    for paper in papers:
        existing = None
        for candidate in unique:
            if _is_same_paper(candidate, paper):
                existing = candidate
                break

        if existing is None:
            unique.append(paper)
        else:
            better = _choose_better_paper(existing, paper)
            if better is not existing:
                unique[unique.index(existing)] = better

    return unique


def _count_domain_matches(text: str) -> int:
    return sum(1 for term in DOMAIN_BOOST_TERMS if term in text)


def _has_low_relevance(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in LOW_RELEVANCE_PATTERNS)


def _compute_paper_score(paper: Dict[str, Any], query: str, entities: List[str], source: str) -> float:
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    base_score = compute_relevance_score(title=title, abstract=abstract, query=query, entities=entities)

    text = _normalize_text(f"{title} {abstract}")
    domain_matches = _count_domain_matches(text)
    low_relevance = _has_low_relevance(text)

    score = base_score + (domain_matches * 0.7)
    if source == "core":
        score += 0.35
    if paper.get("pdf_url"):
        score += 0.2
    if paper.get("open_access"):
        score += 0.15
    if low_relevance:
        score -= 1.5

    exact_entity_hits = sum(1 for entity in entities if entity and entity in text)
    score += exact_entity_hits * 0.4

    return max(score, 0.0)


def hybrid_retrieve(
    query: str,
    expanded_queries: List[str],
    query_intent: Optional[Dict[str, Any]] = None,
    max_results_per_source: int = 6,
    max_final: int = 16,
) -> Dict[str, Any]:
    if query_intent is None:
        query_intent = {}

    rate_limit_mgr = get_rate_limit_manager()
    source_counts = defaultdict(int)
    all_papers: List[Dict[str, Any]] = []

    search_queries = [query] + [q for q in expanded_queries if q != query]
    search_queries = search_queries[:6]

    openalex_enabled = rate_limit_mgr.is_enabled("openalex")
    core_enabled = rate_limit_mgr.is_enabled("core")
    disabled_sources = [source for source, enabled in (("openalex", openalex_enabled), ("core", core_enabled)) if not enabled]

    if disabled_sources:
        print(f"[hybrid_retrieval] disabled sources: {disabled_sources}. Will rely on cached results where available.")

    for search_query in search_queries:
        openalex_papers = []
        core_papers = []

        if rate_limit_mgr.is_enabled("openalex"):
            try:
                openalex_papers = fetch_openalex_papers(search_query, max_results=max_results_per_source)
            except Exception as exc:
                print(f"[hybrid_retrieval] OpenAlex fetch failed for query '{search_query}': {exc}")
                openalex_papers = []

        if rate_limit_mgr.is_enabled("core"):
            try:
                core_papers = fetch_core_papers(search_query, max_results=max_results_per_source)
            except Exception as exc:
                print(f"[hybrid_retrieval] CORE fetch failed for query '{search_query}': {exc}")
                core_papers = []

        for paper in openalex_papers:
            paper["source"] = paper.get("source", "openalex")
        for paper in core_papers:
            paper["source"] = paper.get("source", "core")

        all_papers.extend(openalex_papers)
        all_papers.extend(core_papers)
        source_counts["openalex"] += len(openalex_papers)
        source_counts["core"] += len(core_papers)

    unique_papers = deduplicate_papers(all_papers)
    entities = list(dict.fromkeys((query_intent.get("focus_terms", []) + extract_query_entities(query))))

    scored = []
    for paper in unique_papers:
        score = _compute_paper_score(paper, query, entities, paper.get("source", "openalex"))
        scored.append({"paper": paper, "score": score})

    scored.sort(key=lambda item: item["score"], reverse=True)
    selected = [item["paper"] for item in scored if item["score"] > 0.0][:max_final]

    if not selected:
        selected = [item["paper"] for item in scored[: min(max_final, len(scored))]]

    return {
        "papers": selected,
        "source_counts": dict(source_counts),
        "candidate_count": len(unique_papers),
        "scored_candidates": [{"paper_id": item["paper"].get("paper_id"), "score": round(item["score"], 3)} for item in scored],
        "disabled_sources": disabled_sources,
        "source_health": rate_limit_mgr.get_status(),
    }
