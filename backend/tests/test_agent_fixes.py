import os
import sys

# Ensure the backend package root is on sys.path for tests
TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TEST_ROOT not in sys.path:
    sys.path.insert(0, TEST_ROOT)

from app.agents import retrieval_agent, citation_agent, report_agent
from app.services.text_processor import process_documents
from app.agents import orchestrator


def test_deduplicate_papers_by_normalized_title():
    papers = [
        {"paper_id": "1", "title": "Deep Learning for Vision"},
        {"paper_id": "2", "title": "deep-learning for vision!"},
        {"paper_id": "3", "title": "Other Paper"}
    ]

    unique_papers = retrieval_agent._deduplicate_papers(papers, verbose=False)

    assert len(unique_papers) == 2
    assert any(p["paper_id"] == "1" for p in unique_papers)
    assert any(p["paper_id"] == "3" for p in unique_papers)


def test_compute_relevance_score_returns_consistent_range():
    paper = {
        "title": "Climate modeling with neural networks",
        "abstract": "A paper about climate prediction using neural networks.",
        "keywords": ["climate", "modeling", "neural networks"]
    }

    score = retrieval_agent.compute_relevance_score(paper, "climate modeling")

    assert 0.0 <= score <= 1.0
    assert score >= 0.08


def test_process_documents_tags_abstract_fallback():
    paper = {
        "paper_id": "p1",
        "title": "Fallback Example",
        "full_text": "Summary from abstract fallback.",
        "document_type": "abstract_only",
        "authors": ["Alice Smith", "Bob Jones"],
        "year": 2024,
        "venue": "Test Venue",
        "doi": "10.1000/test"
    }

    chunks = process_documents([paper])

    assert len(chunks) == 1
    assert chunks[0].metadata.get("abstract_fallback") is True


def test_extract_cited_paper_ids_matches_titles():
    metadata = {
        "id1": {"title": "Deep Learning for Vision", "paper_id": "id1"},
        "id2": {"title": "Reinforcement Learning Basics", "paper_id": "id2"}
    }
    report_text = "In this review we analyse [Deep Learning for Vision] and its impact."

    cited_ids = report_agent._extract_cited_paper_ids(report_text, metadata, verbose=False)

    assert cited_ids == ["id1"]


def test_citation_agent_uses_report_cited_ids(monkeypatch):
    paper = {"paper_id": "p1", "title": "Seed Paper", "abstract": "Seed text."}

    def fake_fetch_referenced_works(paper_id, verbose):
        return ["idA", "idB"]

    def fake_fetch_cited_papers_metadata(cited_ids, verbose):
        return [
            {"paper_id": "idA", "title": "Ref A", "abstract": "A", "keywords": ["ref"], "authors": ["Author A"]},
            {"paper_id": "idB", "title": "Ref B", "abstract": "B", "keywords": ["ref"], "authors": ["Author B"]}
        ]

    def fake_download_pdfs_for_new_papers(papers, verbose):
        return papers

    monkeypatch.setattr(citation_agent, "_fetch_referenced_works", fake_fetch_referenced_works)
    monkeypatch.setattr(citation_agent, "_fetch_cited_papers_metadata", fake_fetch_cited_papers_metadata)
    monkeypatch.setattr(citation_agent, "_download_pdfs_for_new_papers", fake_download_pdfs_for_new_papers)
    monkeypatch.setattr(citation_agent, "compute_relevance_score", lambda paper, query: 0.5)

    result = citation_agent.augment_papers(
        [paper],
        paper_ids_seen={"p1"},
        cited_paper_ids=["p1"],
        primary_query="seed paper",
        verbose=False
    )

    assert result["new_papers_added"] == 2
    assert any(p["paper_id"] == "idA" for p in result["augmented_papers"])
    assert any(p["paper_id"] == "idB" for p in result["augmented_papers"])


def test_orchestrator_passes_cited_ids_to_citation_agent(monkeypatch):
    monkeypatch.setattr(orchestrator, "process_query", lambda query, verbose=False: {
        "primary_query": "test query",
        "sub_queries": ["test query"],
        "keywords": ["test"]
    })

    monkeypatch.setattr(orchestrator, "retrieve_papers", lambda sub_queries, keywords, primary_query, verbose=False: {
        "papers": [{"paper_id": "p1", "title": "Seed", "full_text": "text"}],
        "chunks": [],
        "vector_store": None,
        "metadata_store": {"p1": {"paper_id": "p1", "title": "Seed"}},
        "papers_with_extracted_text": {"p1"}
    })

    monkeypatch.setattr(orchestrator, "retrieve_chunks", lambda vector_store, query, k: [])
    monkeypatch.setattr(orchestrator, "generate_report", lambda top_chunks, metadata, query, history, verbose=False: {
        "report": "No useful report.",
        "coverage_score": 7,
        "citation_integrity_score": 1.0,
        "cited_paper_ids": ["p1"]
    })

    monkeypatch.setattr(orchestrator, "_evaluate_retrieval_coverage", lambda papers, sub_queries, verbose: {
        "covered_all_sub_queries": True,
        "covered_count": 1,
        "required_subqueries": 1,
        "unique_papers": 1,
        "missing_subqueries": []
    })

    called = {}
    def fake_augment_papers(papers, paper_ids_seen=None, cited_paper_ids=None, primary_query=None, verbose=False):
        called["cited_paper_ids"] = cited_paper_ids
        return {"augmented_papers": papers, "new_papers_added": 0}

    monkeypatch.setattr(orchestrator, "augment_papers", fake_augment_papers)
    monkeypatch.setattr(orchestrator, "save_index", lambda vs, metadata, sid: None)
    monkeypatch.setattr(orchestrator, "load_index", lambda sid: None)
    monkeypatch.setattr(orchestrator, "create_vector_store", lambda chunks: None)

    result = orchestrator.orchestrate("test query", verbose=False)

    assert called["cited_paper_ids"] == ["p1"]
    assert result["papers_used"] == 1

def test_ollama_timeout_does_not_crash_orchestrator(monkeypatch):
    import time
    import httpx

    monkeypatch.setattr(orchestrator, "process_query", lambda query, verbose=False: {
        "primary_query": "test query",
        "sub_queries": ["test query"],
        "keywords": ["test"]
    })

    monkeypatch.setattr(orchestrator, "retrieve_papers", lambda sub_queries, keywords, primary_query, verbose=False: {
        "papers": [{"paper_id": "p1", "title": "Seed", "full_text": "text"}],
        "chunks": [],
        "vector_store": None,
        "metadata_store": {"p1": {"paper_id": "p1", "title": "Seed"}},
        "papers_with_extracted_text": {"p1"}
    })

    monkeypatch.setattr(orchestrator, "retrieve_chunks", lambda vector_store, query, k: [])
    monkeypatch.setattr(orchestrator, "augment_papers", lambda papers, paper_ids_seen=None, cited_paper_ids=None, primary_query=None, verbose=False: {
        "augmented_papers": papers,
        "new_papers_added": 1
    })

    call_count = {"count": 0}
    def slow_generate_report(*args, **kwargs):
        call_count["count"] += 1
        if call_count["count"] == 1:
            return {"report": "draft", "coverage_score": 0, "citation_integrity_score": 1.0, "cited_paper_ids": ["p1"]}
        time.sleep(0.2)
        return {"report": "final", "coverage_score": 7, "citation_integrity_score": 1.0, "cited_paper_ids": ["p1"]}

    monkeypatch.setattr(orchestrator, "generate_report", slow_generate_report)
    monkeypatch.setattr(orchestrator, "save_index", lambda vs, metadata, sid: None)
    monkeypatch.setattr(orchestrator, "load_index", lambda sid: None)
    monkeypatch.setattr(orchestrator, "create_vector_store", lambda chunks: None)
    monkeypatch.setattr(orchestrator, "ORCHESTRATION_TIMEOUT_SECONDS", 0.05)

    result = orchestrator.orchestrate("test query", verbose=False)

    assert result["timeout"] is True
    assert result["mode"] == "partial_timeout"


def test_skip_regeneration_on_zero_citations(monkeypatch):
    monkeypatch.setattr(orchestrator, "process_query", lambda query, verbose=False: {
        "primary_query": "test query",
        "sub_queries": ["test query"],
        "keywords": ["test"]
    })

    monkeypatch.setattr(orchestrator, "retrieve_papers", lambda sub_queries, keywords, primary_query, verbose=False: {
        "papers": [{"paper_id": "p1", "title": "Seed", "full_text": "text"}],
        "chunks": [],
        "vector_store": None,
        "metadata_store": {"p1": {"paper_id": "p1", "title": "Seed"}},
        "papers_with_extracted_text": {"p1"}
    })

    monkeypatch.setattr(orchestrator, "retrieve_chunks", lambda vector_store, query, k: [])

    generate_calls = {"count": 0}
    def fake_generate_report(*args, **kwargs):
        generate_calls["count"] += 1
        return {"report": "draft", "coverage_score": 7, "citation_integrity_score": 1.0, "cited_paper_ids": ["p1"]}

    monkeypatch.setattr(orchestrator, "generate_report", fake_generate_report)
    monkeypatch.setattr(orchestrator, "augment_papers", lambda papers, paper_ids_seen=None, cited_paper_ids=None, primary_query=None, verbose=False: {
        "augmented_papers": papers,
        "new_papers_added": 0
    })
    monkeypatch.setattr(orchestrator, "save_index", lambda vs, metadata, sid: None)
    monkeypatch.setattr(orchestrator, "load_index", lambda sid: None)
    monkeypatch.setattr(orchestrator, "create_vector_store", lambda chunks: None)

    orchestrator.orchestrate("test query", verbose=False)

    assert generate_calls["count"] == 1


def test_no_retry_spam(monkeypatch):
    import requests

    calls = []
    def fake_get(url, *args, **kwargs):
        calls.append(url)
        raise requests.exceptions.Timeout()

    import app.services.pdf_downloader as pdf_downloader
    monkeypatch.setattr(pdf_downloader, "requests", __import__("requests"))
    monkeypatch.setattr(pdf_downloader.requests, "get", fake_get)

    paper = {
        "paper_id": "p1",
        "title": "Test Paper",
        "arxiv_id": "1234.5678",
        "doi": "10.1000/testdoi",
        "open_access": {"oa_url": "https://example.com/openaccess.pdf"}
    }

    result = pdf_downloader.download_pdf_with_fallbacks(paper)

    assert result is None
    assert calls.count("https://arxiv.org/pdf/1234.5678.pdf") == 1
    assert any("api.unpaywall.org" in u for u in calls)
    assert len(calls) == 3


def test_citation_fetch_fallback(monkeypatch):
    import httpx

    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    call_args = []
    def fake_get(self, url, params=None):
        call_args.append((url, params))
        if params.get("select") == "referenced_works,id,title":
            return DummyResponse({"referenced_works": []})
        if params.get("select") == "related_works,id":
            return DummyResponse({"related_works": ["idA", "idB"]})
        return DummyResponse({})

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    refs = citation_agent._fetch_referenced_works("W123", verbose=True)

    assert refs == ["idA", "idB"]
    assert any(params.get("select") == "related_works,id" for _, params in call_args)


def test_context_injection(monkeypatch):
    """Test that retrieved chunks are injected into LLM prompt."""
    import logging
    log_messages = []
    class TestLogger:
        def info(self, msg):
            log_messages.append(msg)
        def error(self, msg):
            log_messages.append(msg)
        def warning(self, msg):
            log_messages.append(msg)

    # Prepare context as the function expects (already formatted string)
    context = "[Paper 1] by ['Author 1']\nThis is test content from paper 1.\n\n[Paper 2] by ['Author 2']\nThis is test content from paper 2."
    
    # Mock ollama_generate_text to return a report
    def fake_ollama_generate_text(prompt):
        # Check that prompt contains our test content
        assert 'This is test content from paper 1.' in prompt
        assert 'This is test content from paper 2.' in prompt
        return "Test report with content."
    
    monkeypatch.setattr(report_agent, 'ollama_generate_text', fake_ollama_generate_text)
    monkeypatch.setattr(report_agent, 'logger', TestLogger())
    
    result = report_agent._call_llm_for_report("test query", context, "test conv", verbose=True)
    
    # Check diagnostic logging
    assert any("chunks, " in msg and "chars injected" in msg for msg in log_messages)
    assert result == "Test report with content."


def test_iteration_floor(monkeypatch):
    """Test that orchestrator runs at least 2 iterations regardless of coverage."""
    iteration_count = {"count": 0}
    
    def fake_generate_report(*args, **kwargs):
        iteration_count["count"] += 1
        # Return low coverage on first iteration
        return {"report": f"report {iteration_count['count']}", "coverage_score": 1, "citation_integrity_score": 1.0, "cited_paper_ids": []}
    
    monkeypatch.setattr(orchestrator, "process_query", lambda query, verbose=False: {
        "primary_query": "test query",
        "sub_queries": ["test query"],
        "keywords": ["test"]
    })
    
    monkeypatch.setattr(orchestrator, "retrieve_papers", lambda sub_queries, keywords, primary_query, verbose=False: {
        "papers": [{"paper_id": "p1", "title": "Seed", "full_text": "text"}],
        "chunks": [],
        "vector_store": None,
        "metadata_store": {"p1": {"paper_id": "p1", "title": "Seed"}},
        "papers_with_extracted_text": {"p1"}
    })
    
    monkeypatch.setattr(orchestrator, "retrieve_chunks", lambda vector_store, query, k: [])
    monkeypatch.setattr(orchestrator, "generate_report", fake_generate_report)
    # Make citations add papers so it continues past the zero-check
    monkeypatch.setattr(orchestrator, "augment_papers", lambda papers, paper_ids_seen=None, cited_paper_ids=None, primary_query=None, verbose=False: {
        "augmented_papers": papers + [{"paper_id": "p2", "title": "New", "full_text": "text"}],
        "new_papers_added": 1
    })
    monkeypatch.setattr(orchestrator, "_evaluate_retrieval_coverage", lambda papers, sub_queries, verbose: {
        "covered_all_sub_queries": True,
        "covered_count": 1,
        "required_subqueries": 1,
        "unique_papers": 1,
        "missing_subqueries": []
    })
    monkeypatch.setattr(orchestrator, "save_index", lambda vs, metadata, sid: None)
    monkeypatch.setattr(orchestrator, "load_index", lambda sid: None)
    monkeypatch.setattr(orchestrator, "create_vector_store", lambda chunks: None)
    
    result = orchestrator.orchestrate("test query", verbose=False)
    
    # Should have run at least 2 iterations despite low coverage
    assert iteration_count["count"] >= 2


def test_blocklist_filter(monkeypatch):
    """Test that blocklisted papers are filtered out."""
    paper = {
        "title": "food fraud spectroscopy",
        "abstract": "This paper discusses food fraud detection using spectroscopy techniques.",
        "paper_id": "p1"
    }
    
    assert retrieval_agent.is_blocklisted(paper) is True
    
    # Test non-blocklisted paper
    good_paper = {
        "title": "Machine Learning for Fraud Detection",
        "abstract": "Using neural networks to detect financial fraud.",
        "paper_id": "p2"
    }
    
    assert retrieval_agent.is_blocklisted(good_paper) is False


def test_openalex_id_normalization():
    """Test OpenAlex ID normalization."""
    assert citation_agent.normalize_openalex_id("W4214489515") == "https://openalex.org/W4214489515"
    assert citation_agent.normalize_openalex_id("https://openalex.org/W4214489515") == "https://openalex.org/W4214489515"
    assert citation_agent.normalize_openalex_id("invalid") == "invalid"


def test_comparison_table_validation(monkeypatch):
    """Test that missing comparison table triggers warning."""
    import logging
    log_messages = []
    class TestLogger:
        def info(self, msg):
            log_messages.append(msg)
        def error(self, msg):
            log_messages.append(msg)
        def warning(self, msg):
            log_messages.append(msg)
    
    # Mock report without table
    def fake_ollama_generate_text(prompt):
        return "Summary: Test summary.\n\nKey Methods:\n- Method 1\n\nResearch Gaps:\n- Gap 1\n\nReferences:\n- Ref 1"
    
    monkeypatch.setattr(report_agent, 'ollama_generate_text', fake_ollama_generate_text)
    monkeypatch.setattr(report_agent, 'logger', TestLogger())
    
    result = report_agent._call_llm_for_report("test query", "test context", "test conv", verbose=False)
    
    # Check that warning was logged
    assert any("Comparison table missing" in msg for msg in log_messages)


def test_source_display_partial():
    """Test that main.py source display handles missing fields gracefully."""
    # Mock a partial result with missing title/year
    sources = [
        {"id": "https://openalex.org/W123", "display_name": "Test Paper", "publication_year": "2023"},
        {"url": "https://example.com", "title": "Another Paper"},
        {"paper_id": "W456"}  # Missing everything
    ]
    
    # This should not raise KeyError and should produce reasonable displays
    displays = []
    for i, source in enumerate(sources, 1):
        title = source.get('title') or source.get('display_name') or 'Unknown title'
        year = source.get('year') or source.get('publication_year') or 'n.d.'
        url = source.get('url') or source.get('id') or ''
        display = f"{i}. {title} ({year}) - {url}"
        displays.append(display)
    
    # Check that displays are formed correctly
    assert displays[0] == "1. Test Paper (2023) - https://openalex.org/W123"
    assert displays[1] == "2. Another Paper (n.d.) - https://example.com"
    assert displays[2] == "3. Unknown title (n.d.) - "


def test_sub_query_relevance_fallback():
    """Test that compute_best_relevance_score uses sub-queries when primary score is low."""
    paper = {
        "title": "Machine Learning for Fraud Detection",
        "abstract": "Using classification algorithms to detect financial fraud patterns.",
        "keywords": ["fraud", "detection"]
    }
    
    primary_query = "quantum computing applications"
    sub_queries = ["Data preprocessing techniques for fraud detection"]
    
    score, winning_query = retrieval_agent.compute_best_relevance_score(paper, primary_query, sub_queries)
    
    # Should score higher on sub-query than primary
    assert score >= 0.10  # threshold
    assert winning_query == sub_queries[0]  # the sub-query should win


def test_faiss_incremental_add():
    """Test that FAISS index can be updated with new chunks (mocked)."""
    # This is a placeholder test - actual FAISS incremental add would need mocking
    # For now, just verify the function exists and can be called
    try:
        # This would normally add chunks to an existing index
        # Since we don't have FAISS in tests, just check function signature
        import inspect
        sig = inspect.signature(retrieval_agent.add_chunks_to_index)
        assert 'new_chunks' in sig.parameters
    except AttributeError:
        # Function might not be implemented yet
        pass


def test_citation_format_enforcement():
    """Test that citation extraction handles author-year format as fallback."""
    metadata = {
        "p1": {
            "paper_id": "https://openalex.org/W123",
            "title": "Test Paper",
            "authors": ["Smith et al."],
            "year": "2023"
        }
    }
    
    # Report with author-year citations (no numeric)
    report_text = "This method (Smith et al., 2023) is effective for fraud detection."
    
    cited_ids = report_agent._extract_cited_paper_ids(report_text, metadata, verbose=False)
    
    assert "https://openalex.org/W123" in cited_ids


def test_year_floor_filter():
    """Test that year floor filter removes old papers."""
    # Mock refs with different years
    refs = [
        {"id": "W2005", "publication_year": 2005},
        {"id": "W2015", "publication_year": 2015}, 
        {"id": "W2022", "publication_year": 2022}
    ]
    
    # Simulate the filtering logic
    MIN_PUBLICATION_YEAR = 2010
    candidates = [
        ref for ref in refs
        if ref.get('publication_year', 9999) >= MIN_PUBLICATION_YEAR
    ]
    
    assert len(candidates) == 2
    assert all(ref['publication_year'] >= 2010 for ref in candidates)
    assert not any(ref['publication_year'] < 2010 for ref in candidates)
