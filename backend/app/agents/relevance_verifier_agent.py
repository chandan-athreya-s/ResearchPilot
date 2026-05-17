from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

from app.agents.base_agent import BaseAgent
from app.core.state import ResearchState
from app.services.query_analyzer import analyze_query

METHOD_KEYWORDS = {
    "method", "algorithm", "approach", "architecture", "framework",
    "system", "model", "training", "evaluation", "benchmark", "implementation",
    "deployment", "fine tuning", "fine-tuning", "agentic", "workflow"
}

RELEVANCE_KEYWORDS = {
    "retrieval augmented generation": ["rag", "retrieval augmented generation"],
    "fine tuning": ["fine tuning", "fine-tuning", "fine tune"],
    "large language model": ["llm", "large language model", "large language models"],
    "agentic workflows": ["agentic workflows", "agentic workflow", "autonomous agents"],
    "enterprise knowledge systems": ["enterprise knowledge systems", "enterprise ai systems", "knowledge management systems"],
}

LOW_RELEVANCE_PATTERNS = [
    r"\bbiomedical\b",
    r"\bfederated learning\b",
    r"\bxai\b",
    r"\bexplainable ai\b",
    r"\bsurvey\b",
    r"\bmedical\b",
    r"\bhealthcare\b",
]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_query_entities(query: str) -> List[str]:
    """Extract clean entities from the query for relevance scoring."""
    analysis = analyze_query(query)
    entities = []

    entities.extend(analysis.get("focus_terms", []))
    for pair in analysis.get("comparison_pairs", []):
        entities.extend(pair)

    suffix_match = re.search(r"\b(?:for|in|within|using|across)\s+(.+)$", query, re.IGNORECASE)
    if suffix_match:
        suffix = _normalize_text(suffix_match.group(1))
        if suffix and suffix not in entities:
            entities.append(suffix)

    normalized_entities = []
    seen = set()
    for entity in entities:
        normalized = _normalize_text(entity)
        if not normalized:
            continue
        normalized = _normalize_entity_alias(normalized)
        if normalized not in seen:
            seen.add(normalized)
            normalized_entities.append(normalized)

    return normalized_entities


def _normalize_entity_alias(entity: str) -> str:
    entity = entity.strip().lower()
    for canonical, aliases in RELEVANCE_KEYWORDS.items():
        if entity == canonical or entity in aliases:
            return canonical
        if any(alias in entity for alias in aliases):
            return canonical
    return entity


def compute_semantic_overlap(text: str, entities: List[str]) -> float:
    """Compute a lightweight semantic overlap score using entity and keyword matches."""
    text = _normalize_text(text)
    if not text or not entities:
        return 0.0

    overlap = 0.0
    token_set = set(text.split())
    for entity in entities:
        normalized = _normalize_text(entity)
        if not normalized:
            continue
        if normalized in text:
            overlap += 2.0
        else:
            entity_tokens = normalized.split()
            hit_tokens = sum(1 for token in entity_tokens if token in token_set)
            overlap += hit_tokens / max(len(entity_tokens), 1)

    # Add fuzzy similarity for the full query and text if not enough exact matches
    if overlap < 1.0:
        overlap += SequenceMatcher(None, text, " ".join(entities)).ratio()

    return overlap


def compute_relevance_score(title: str, abstract: str, query: str, entities: List[str]) -> float:
    """Compute a composite relevance score for a paper based on title/abstract and query entities."""
    title_text = _normalize_text(title or "")
    abstract_text = _normalize_text(abstract or "")
    query_entities = list(dict.fromkeys(entities))

    title_overlap = compute_semantic_overlap(title_text, query_entities)
    abstract_overlap = compute_semantic_overlap(abstract_text, query_entities)

    exact_keywords = sum(
        1 for entity in query_entities if entity and (entity in title_text or entity in abstract_text)
    )

    title_relevance = 2.0 if any(entity in title_text for entity in query_entities) else 0.0
    method_signal = 1.0 if any(keyword in abstract_text or keyword in title_text for keyword in METHOD_KEYWORDS) else 0.0

    low_relevance_penalty = 0.0
    for pattern in LOW_RELEVANCE_PATTERNS:
        if re.search(pattern, title_text) or re.search(pattern, abstract_text):
            low_relevance_penalty += 1.5

    score = (
        title_overlap * 3.0
        + abstract_overlap * 2.0
        + exact_keywords * 2.0
        + title_relevance * 1.5
        + method_signal * 1.5
        - low_relevance_penalty
    )

    if exact_keywords == 0 and title_relevance == 0 and method_signal == 0:
        return 0.0
    if exact_keywords == 0 and title_relevance == 0:
        score = max(score - 4.0, 0.0)

    return max(score, 0.0)


def filter_irrelevant_papers(papers: List[Dict[str, Any]], query: str, threshold: float = 4.0) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Filter out papers that are weakly aligned to the query."""
    entities = extract_query_entities(query)
    scored = []
    for paper in papers:
        score = compute_relevance_score(
            title=paper.get("title", ""),
            abstract=paper.get("abstract", ""),
            query=query,
            entities=entities,
        )
        scored.append({
            "paper": paper,
            "score": score,
            "title_matches": compute_semantic_overlap(paper.get("title", ""), entities),
            "abstract_matches": compute_semantic_overlap(paper.get("abstract", ""), entities),
        })

    scored.sort(key=lambda item: item["score"], reverse=True)
    filtered = [item["paper"] for item in scored if item["score"] >= threshold]
    if not filtered and scored:
        filtered = [item["paper"] for item in scored[: min(4, len(scored))]]

    return filtered, scored


class RelevanceVerifierAgent(BaseAgent):
    """Reject semantically weak or tangential papers before acquisition."""

    name = "RelevanceVerifierAgent"

    def run(self, state: ResearchState) -> ResearchState:
        try:
            if not state.papers:
                self._log("No papers available for relevance verification.")
                return state

            query_entities = extract_query_entities(state.query)
            relevant_papers, scored = filter_irrelevant_papers(state.papers, state.query)

            rejected_count = max(0, len(state.papers) - len(relevant_papers))
            state.papers = relevant_papers
            state.metadata_store = {paper["paper_id"]: paper.copy() for paper in relevant_papers}
            state.diagnostics["papers_rejected"] = rejected_count
            state.diagnostics["relevance_scores"] = [
                {"paper_id": item["paper"].get("paper_id"), "score": round(item["score"], 3)}
                for item in scored
            ]
            state.diagnostics["entity_matches"] = {
                item["paper"].get("paper_id"): {
                    "title": round(item["title_matches"], 3),
                    "abstract": round(item["abstract_matches"], 3),
                    "overall_score": round(item["score"], 3),
                }
                for item in scored
            }

            self._log(
                f"Verified relevance for {len(state.papers)} papers; rejected {rejected_count} weak papers"
            )
        except Exception as error:
            error_message = f"Relevance verification failed: {error}"
            self._log(error_message)
            state.errors.append(error_message)
        return state
