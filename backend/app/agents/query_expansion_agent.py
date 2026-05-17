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


def generate_subqueries(query: str) -> List[str]:
    """Generate a set of targeted search queries from the original comparison or survey query."""
    analysis = analyze_query(query)
    normalized_entities = normalize_entities(
        analysis.get("focus_terms", []) + [item for pair in analysis.get("comparison_pairs", []) for item in pair]
    )
    query_text = _normalize_text(query)
    subqueries: List[str] = []
    seen: set[str] = set()

    def add_query(item: str) -> None:
        item = _normalize_text(item)
        if item and item not in seen:
            seen.add(item)
            subqueries.append(item)

    add_query(query_text)

    for entity in normalized_entities:
        add_query(entity)
        add_query(f"{entity} research")
        for anchor in DOMAIN_ANCHORS:
            if anchor not in entity:
                add_query(f"{entity} {anchor}")
        for expansion in ENTITY_EXPANSIONS.get(entity, []):
            add_query(expansion)

    if analysis.get("query_type") == "comparison":
        for left, right in analysis.get("comparison_pairs", []):
            left_norm = _normalize_text(left)
            right_norm = _normalize_text(right)
            add_query(f"{left_norm} vs {right_norm}")
            add_query(f"{left_norm} and {right_norm} comparison")
            if "enterprise" in query_text:
                add_query(f"{left_norm} vs {right_norm} enterprise ai")
            if "retrieval" in query_text or "rag" in query_text:
                add_query(f"{left_norm} vs {right_norm} retrieval systems")

    if not normalized_entities and query_text:
        add_query(query_text)

    # Keep the most targeted expansions first, limit noise
    if len(subqueries) > 10:
        subqueries = subqueries[:10]

    return subqueries


class QueryExpansionAgent(BaseAgent):
    """Generate query variants with entity normalization and targeted academic search phrases."""

    name = "QueryExpansionAgent"

    def run(self, state: ResearchState) -> ResearchState:
        try:
            state.expanded_queries = generate_subqueries(state.query)
            if not state.expanded_queries:
                state.expanded_queries = [state.query]

            state.diagnostics["expanded_queries"] = state.expanded_queries
            state.diagnostics["normalized_entities"] = normalize_entities(
                analyze_query(state.query).get("focus_terms", [])
                + [item for pair in analyze_query(state.query).get("comparison_pairs", []) for item in pair]
            )
            self._log(f"Expanded query into {len(state.expanded_queries)} targeted searches")
        except Exception as error:
            error_message = f"Query expansion failed: {error}"
            self._log(error_message)
            state.errors.append(error_message)
        return state
