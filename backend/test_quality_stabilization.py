import sys
from pathlib import Path
from unittest.mock import patch

workspace_root = Path(__file__).resolve().parent
sys.path.insert(0, str(workspace_root))

from app.agents.query_expansion_agent import expand_query, generate_subqueries, normalize_entities
from app.agents.relevance_verifier_agent import extract_query_entities, filter_irrelevant_papers
from app.services.llm_service import remove_placeholder_citations, post_process_citations
from app.services.openalex_client import build_search_query
from app.agents.compression_agent import compress_chunk


def test_query_entity_extraction_and_normalization():
    query = "Comparison of retrieval augmented generation, fine tuning, and agentic workflows for enterprise knowledge management systems"
    entities = extract_query_entities(query)

    assert "retrieval augmented generation" in entities
    assert "fine tuning" in entities
    assert "agentic workflows" in entities
    assert "enterprise knowledge systems" in entities


def test_relevance_verifier_filters_irrelevant_papers():
    query = "Comparative study of retrieval augmented generation and fine tuning for enterprise AI systems"
    papers = [
        {
            "paper_id": "p1",
            "title": "Retrieval Augmented Generation for Knowledge Systems",
            "abstract": "We propose a RAG architecture that improves enterprise knowledge access.",
        },
        {
            "paper_id": "p2",
            "title": "Federated Learning for Medical Imaging",
            "abstract": "This paper focuses on privacy-preserving models in healthcare.",
        },
    ]

    filtered, scored = filter_irrelevant_papers(papers, query, threshold=4.0)
    assert len(filtered) == 1
    assert filtered[0]["paper_id"] == "p1"
    assert scored[0]["score"] >= scored[1]["score"]


def test_placeholder_citation_removal_and_post_processing():
    answer = "This approach is backed by evidence [N123] and later by Source 1. [Source 1, Chunk 2]"
    source_references = {"Source 1": {"authors": ["Smith"], "year": 2024}}
    old_to_new_ref_num = {1: 1}

    cleaned_answer, stats = post_process_citations(answer, source_references, old_to_new_ref_num)
    assert "[N123]" not in cleaned_answer
    assert "[1]" in cleaned_answer
    assert stats["placeholder_removed"] == 1
    assert stats["replaced"] == 2


def test_openalex_query_expansion_includes_normalized_terms():
    query = "Compare RAG vs fine-tuning in enterprise knowledge systems"
    expanded = build_search_query(query)
    assert "retrieval augmented generation" in expanded
    assert "fine tuning" in expanded
    assert "enterprise knowledge systems" in expanded


def test_query_expansion_agent_generates_targeted_subqueries():
    query = "Comparison of retrieval augmented generation, fine tuning, and agentic workflows for enterprise knowledge systems"
    subqueries = generate_subqueries(query)

    assert any("retrieval augmented generation" in q for q in subqueries)
    assert any("fine tuning" in q for q in subqueries)
    assert any("agentic workflows" in q for q in subqueries)
    assert any("enterprise knowledge systems" in q for q in subqueries)
    assert len(subqueries) >= 4


def test_normalize_entities_handles_aliases():
    entities = ["RAG", "fine-tuning", "LLM", "enterprise ai systems"]
    normalized = normalize_entities(entities)
    assert "retrieval augmented generation" in normalized
    assert "fine tuning" in normalized
    assert "large language model" in normalized
    assert "enterprise knowledge systems" in normalized


def test_hybrid_retrieval_merges_openalex_and_core_candidates():
    from app.services.hybrid_retrieval import hybrid_retrieve

    def fake_openalex(query, max_results=8):
        return [
            {
                "paper_id": "o1",
                "title": "Retrieval Augmented Generation for Enterprise Systems",
                "abstract": "A study of RAG and enterprise knowledge systems.",
                "doi": "10.1000/openalex1",
                "pdf_url": "http://example.com/openalex.pdf",
                "open_access": {},
                "source": "openalex",
            }
        ]

    def fake_core(query, max_results=8):
        return [
            {
                "paper_id": "c1",
                "title": "Fine tuning large language models for enterprise AI",
                "abstract": "Fine tuning LLMs in retrieval augmented workflows.",
                "doi": "10.1000/core1",
                "pdf_url": "http://example.com/core.pdf",
                "open_access": {},
                "source": "core",
            }
        ]

    with patch("app.services.hybrid_retrieval.fetch_openalex_papers", new=fake_openalex), \
         patch("app.services.hybrid_retrieval.fetch_core_papers", new=fake_core):
        result = hybrid_retrieve(
            query="Compare RAG vs fine tuning for enterprise AI",
            expanded_queries=["retrieval augmented generation enterprise systems", "fine tuning large language models"],
            query_intent={"focus_terms": ["retrieval augmented generation", "fine tuning"]},
            max_results_per_source=2,
            max_final=5,
        )

    assert result["papers"]
    assert result["source_counts"]["openalex"] == 2 or result["source_counts"]["openalex"] == 1
    assert result["source_counts"]["core"] == 2 or result["source_counts"]["core"] == 1
    assert any(p["paper_id"] == "o1" for p in result["papers"])
    assert any(p["paper_id"] == "c1" for p in result["papers"])


def test_compression_agent_prioritizes_technical_content():
    class Doc:
        pass

    doc = Doc()
    doc.page_content = (
        "This paper introduces a novel architecture. "
        "The method uses retrieval augmented generation and achieves 95% accuracy. "
        "In this study, we provide a comprehensive evaluation. "
        "The remainder of the paper is organized as follows."
    )

    compressed = compress_chunk(doc, target_tokens=50)
    assert "retrieval augmented generation" in compressed.page_content.lower()
    assert "architecture" in compressed.page_content.lower() or "accuracy" in compressed.page_content.lower()
    assert len(compressed.page_content.split()) <= 55
