from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


@dataclass
class ResearchState:
    """Shared mutable state carried through agent execution."""

    query: str
    query_intent: Dict[str, Any] = field(default_factory=dict)
    papers: List[Dict[str, Any]] = field(default_factory=list)
    metadata_store: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    documents: List[Any] = field(default_factory=list)
    retrieved_chunks: List[Any] = field(default_factory=list)
    generated_answer: str = ""
    references: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=lambda: {
        "query_type": "",
        "retrieval_time": 0,
        "compression_time": 0,
        "reasoning_time": 0,
        "chunk_count": 0,
        "retrieved_chunk_count": 0,
        "source_count": 0,
        "candidate_chunk_count": 0,
        "compression_ratio": 1.0,
        "compressed_chunk_count": 0,
        "estimated_prompt_tokens": 0,
        "deduplicated_chunks": 0,
        "removed_redundancies": 0,
        "focus_coverage": {},
        "candidate_coverage": {},
        "papers_retrieved": 0,
        "papers_rejected": 0,
        "relevance_scores": [],
        "entity_matches": {},
        "grounding_validation": {},
        "citation_cleanup_count": 0,
        "expanded_queries": [],
        "retrieval_source_counts": {},
    })
    errors: List[str] = field(default_factory=list)
    papers_with_extracted_text: Set[str] = field(default_factory=set)
    filtered_papers: List[Dict[str, Any]] = field(default_factory=list)
    expanded_queries: List[str] = field(default_factory=list)
