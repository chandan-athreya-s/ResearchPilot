from __future__ import annotations

import re
from typing import List

from app.agents.base_agent import BaseAgent
from app.core.state import ResearchState
from app.services.query_analyzer import QueryAnalyzer, analyze_query

DOMAIN_ANCHORS = [
    "enterprise systems",
    "enterprise ai",
    "knowledge management systems",
    "autonomous agents",
    "multi-agent systems",
    "retrieval systems",
]

ENTITY_EXPANSIONS = {
    "retrieval augmented generation": [
        "retrieval augmented generation enterprise systems",
        "rag retrieval systems",
        "retrieval augmented generation methodology",
    ],
    "fine tuning": [
        "fine tuning large language models",
        "fine tuning llms",
        "fine tuning evaluation",
    ],
    "large language model": [
        "large language model fine tuning",
        "llm evaluation",
        "large language model retrieval",
    ],
    "agentic workflows": [
        "agentic workflows enterprise assistants",
        "multi-agent workflows",
        "autonomous agent workflows",
    ],
    "enterprise knowledge systems": [
        "enterprise knowledge management systems",
        "enterprise ai systems",
        "enterprise knowledge systems architecture",
    ],
}

NORMALIZATION_MAP = QueryAnalyzer.NORMALIZATION_MAP


def _normalize_text(text: str) -> str:
    text = text or ""
    text = text.strip().lower()
    text = re.sub(r"[\-\/]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_entities(entities: List[str]) -> List[str]:
    """Normalize entities to canonical academic search concepts."""
    normalized = []
    seen = set()

    for entity in entities:
        if not entity:
            continue

        entity_text = _normalize_text(entity)
        aliases = []
        remaining = entity_text

        for alias, canonical in sorted(NORMALIZATION_MAP.items(), key=lambda item: len(item[0]), reverse=True):
            pattern = rf"\b{re.escape(alias)}\b"
            while re.search(pattern, remaining):
                aliases.append(canonical)
                remaining = re.sub(pattern, " ", remaining, count=1)

        remaining = _normalize_text(remaining)
        if remaining and remaining not in aliases:
            aliases.append(remaining)

        for alias in aliases:
            if alias and alias not in seen:
                seen.add(alias)
                normalized.append(alias)

    return normalized


def expand_query(query: str) -> str:
    """Expand the original query with normalized entities and comparison terms."""
    analysis = analyze_query(query)
    entities = normalize_entities(
        analysis.get("focus_terms", []) + [item for pair in analysis.get("comparison_pairs", []) for item in pair]
    )
    tokens = [query.strip()] + entities
    return " ".join(dict.fromkeys([token for token in tokens if token]))


def _get_adaptive_expansion_limit(query_type: str, entity_count: int) -> int:
    """Determine the maximum number of expansion queries based on query type and complexity."""
    base_limits = {
        "general": 2,
        "implementation": 3,
        "challenges": 4,
        "comparison": 4,
        "survey": 5,
    }
    
    limit = base_limits.get(query_type, 3)
    
    # Adjust based on entity complexity
    if entity_count <= 1:
        limit = max(2, limit - 1)
    elif entity_count >= 3:
        limit = min(6, limit + 1)
    
    return limit


def generate_subqueries(query: str) -> List[str]:
    """Generate targeted search queries adaptively based on query type and complexity."""
    analysis = analyze_query(query)
    query_type = analysis.get("query_type", "general")
    
    normalized_entities = normalize_entities(
        analysis.get("focus_terms", []) + [item for pair in analysis.get("comparison_pairs", []) for item in pair]
    )
    
    # Determine adaptive expansion limit
    adaptive_limit = _get_adaptive_expansion_limit(query_type, len(normalized_entities))
    
    query_text = _normalize_text(query)
    subqueries: List[str] = []
    seen: set[str] = set()

    def add_query(item: str) -> None:
        item = _normalize_text(item)
        if item and item not in seen and len(subqueries) < adaptive_limit:
            seen.add(item)
            subqueries.append(item)

    # Always add the original expanded query first (highest priority)
    add_query(query_text)

    # Add entity-based queries
    for entity in normalized_entities:
        add_query(entity)
        if len(subqueries) >= adaptive_limit:
            break
    
    # Add targeted domain queries
    if len(subqueries) < adaptive_limit:
        for entity in normalized_entities:
            add_query(f"{entity} research")
            if len(subqueries) >= adaptive_limit:
                break
    
    # Add domain anchor combinations
    if len(subqueries) < adaptive_limit and query_type in ("comparison", "survey"):
        for entity in normalized_entities:
            for anchor in DOMAIN_ANCHORS:
                if anchor not in entity and len(subqueries) < adaptive_limit:
                    add_query(f"{entity} {anchor}")
    
    # Add entity expansion variants
    if len(subqueries) < adaptive_limit:
        for entity in normalized_entities:
            for expansion in ENTITY_EXPANSIONS.get(entity, []):
                if len(subqueries) < adaptive_limit:
                    add_query(expansion)
    
    # Add comparison-specific queries
    if query_type == "comparison" and len(subqueries) < adaptive_limit:
        for left, right in analysis.get("comparison_pairs", []):
            left_norm = _normalize_text(left)
            right_norm = _normalize_text(right)
            add_query(f"{left_norm} vs {right_norm}")
            if len(subqueries) >= adaptive_limit:
                break
            if "enterprise" in query_text:
                add_query(f"{left_norm} vs {right_norm} enterprise ai")
            if len(subqueries) >= adaptive_limit:
                break
            if "retrieval" in query_text or "rag" in query_text:
                add_query(f"{left_norm} vs {right_norm} retrieval systems")
            if len(subqueries) >= adaptive_limit:
                break

    # Fallback to original query if no expansions
    if not subqueries and query_text:
        subqueries.append(query_text)

    return subqueries


class QueryExpansionAgent(BaseAgent):
    """Generate adaptive query variants based on query type and complexity."""

    name = "QueryExpansionAgent"

    def run(self, state: ResearchState) -> ResearchState:
        try:
            state.expanded_queries = generate_subqueries(state.query)
            if not state.expanded_queries:
                state.expanded_queries = [state.query]

            query_analysis = analyze_query(state.query)
            state.query_intent = query_analysis

            state.diagnostics["expanded_queries"] = state.expanded_queries
            state.diagnostics["expansion_count"] = len(state.expanded_queries)
            state.diagnostics["query_type"] = query_analysis.get("query_type")
            state.diagnostics["normalized_entities"] = normalize_entities(
                query_analysis.get("focus_terms", [])
                + [item for pair in query_analysis.get("comparison_pairs", []) for item in pair]
            )
            self._log(f"Adaptively expanded query into {len(state.expanded_queries)} targeted searches (type: {query_analysis.get('query_type')})")
        except Exception as error:
            error_message = f"Query expansion failed: {error}"
            self._log(error_message)
            state.errors.append(error_message)
        return state
