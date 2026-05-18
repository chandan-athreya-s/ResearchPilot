import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

workspace_root = Path(__file__).resolve().parent
sys.path.insert(0, str(workspace_root))

from app.services.hybrid_retrieval import hybrid_retrieve
from app.services.core_client import fetch_core_papers
from app.services.openalex_client import fetch_papers as fetch_openalex_papers


class TestHybridRetrieval(unittest.TestCase):
    def setUp(self):
        self.openalex_paper = {
            "paper_id": "openalex-1",
            "title": "OpenAlex Retrieval Paper",
            "abstract": "This paper evaluates retrieval methods.",
            "source": "openalex",
        }
        self.core_paper = {
            "paper_id": "core-1",
            "title": "CORE Retrieval Paper",
            "abstract": "This paper evaluates retrieval methods.",
            "source": "core",
        }

    @patch("app.services.hybrid_retrieval._compute_paper_score", return_value=1.0)
    @patch("app.services.hybrid_retrieval.fetch_core_papers", return_value=[])
    @patch("app.services.hybrid_retrieval.fetch_openalex_papers")
    def test_openalex_only_retrieval(self, openalex_fetch, core_fetch, score_mock):
        openalex_fetch.return_value = [self.openalex_paper]

        result = hybrid_retrieve("query", [], {"focus_terms": ["retrieval"]}, max_results_per_source=2, max_final=4)

        self.assertEqual(len(result["papers"]), 1)
        self.assertEqual(result["papers"][0]["paper_id"], "openalex-1")
        self.assertEqual(result["source_counts"].get("openalex"), 1)
        self.assertEqual(result["source_counts"].get("core"), 0)
        self.assertEqual(result["candidate_count"], 1)

    @patch("app.services.hybrid_retrieval._compute_paper_score", return_value=1.0)
    @patch("app.services.hybrid_retrieval.fetch_openalex_papers", return_value=[])
    @patch("app.services.hybrid_retrieval.fetch_core_papers")
    def test_core_only_retrieval(self, core_fetch, openalex_fetch, score_mock):
        core_fetch.return_value = [self.core_paper]

        result = hybrid_retrieve("query", [], {"focus_terms": ["retrieval"]}, max_results_per_source=2, max_final=4)

        self.assertEqual(len(result["papers"]), 1)
        self.assertEqual(result["papers"][0]["paper_id"], "core-1")
        self.assertEqual(result["source_counts"].get("openalex"), 0)
        self.assertEqual(result["source_counts"].get("core"), 1)
        self.assertEqual(result["candidate_count"], 1)

    @patch("app.services.hybrid_retrieval._compute_paper_score", return_value=1.0)
    @patch("app.services.hybrid_retrieval.fetch_core_papers")
    @patch("app.services.hybrid_retrieval.fetch_openalex_papers")
    def test_hybrid_retrieval_with_both_sources(self, openalex_fetch, core_fetch, score_mock):
        openalex_fetch.return_value = [self.openalex_paper]
        core_fetch.return_value = [self.core_paper]

        result = hybrid_retrieve("query", [], {"focus_terms": ["retrieval"]}, max_results_per_source=2, max_final=4)

        paper_ids = {paper["paper_id"] for paper in result["papers"]}
        self.assertEqual(paper_ids, {"openalex-1", "core-1"})
        self.assertEqual(result["source_counts"].get("openalex"), 1)
        self.assertEqual(result["source_counts"].get("core"), 1)
        self.assertEqual(result["candidate_count"], 2)

    @patch("app.services.hybrid_retrieval._compute_paper_score", return_value=1.0)
    @patch("app.services.hybrid_retrieval.fetch_openalex_papers")
    def test_core_malformed_response_does_not_crash(self, openalex_fetch, score_mock):
        openalex_fetch.return_value = [self.openalex_paper]

        bad_response = Mock()
        bad_response.status_code = 200
        bad_response.headers = {"Content-Type": "application/json"}
        bad_response.text = "not a json"
        bad_response.json = Mock(side_effect=ValueError("Expecting value: line 1 column 1 (char 0)"))

        with patch("app.services.core_client.requests.get", return_value=bad_response):
            result = hybrid_retrieve("query", [], {"focus_terms": ["retrieval"]}, max_results_per_source=2, max_final=4)

        self.assertEqual(len(result["papers"]), 1)
        self.assertEqual(result["papers"][0]["paper_id"], "openalex-1")

    @patch("app.services.hybrid_retrieval._compute_paper_score", return_value=1.0)
    @patch("app.services.hybrid_retrieval.fetch_core_papers")
    def test_openalex_rate_limit_response_falls_back_to_core(self, core_fetch, score_mock):
        core_fetch.return_value = [self.core_paper]

        bad_response = Mock()
        bad_response.status_code = 429
        bad_response.headers = {"Content-Type": "text/html"}
        bad_response.text = "<html><body>Rate limit exceeded</body></html>"
        bad_response.json = Mock(side_effect=ValueError("Expecting value: line 1 column 1 (char 0)"))

        with patch("app.services.openalex_client.requests.get", return_value=bad_response):
            result = hybrid_retrieve("query", [], {"focus_terms": ["retrieval"]}, max_results_per_source=2, max_final=4)

        self.assertEqual(len(result["papers"]), 1)
        self.assertEqual(result["papers"][0]["paper_id"], "core-1")
        self.assertEqual(result["source_counts"].get("openalex"), 0)
        self.assertEqual(result["source_counts"].get("core"), 1)

    def test_fetch_core_papers_returns_empty_on_malformed_json(self):
        bad_response = Mock()
        bad_response.status_code = 200
        bad_response.headers = {"Content-Type": "application/json"}
        bad_response.text = "not a json"
        bad_response.json = Mock(side_effect=ValueError("Expecting value: line 1 column 1 (char 0)"))

        with patch("app.services.core_client.requests.get", return_value=bad_response):
            papers = fetch_core_papers("query", max_results=1)

        self.assertEqual(papers, [])

    def test_fetch_openalex_papers_returns_empty_on_empty_body(self):
        bad_response = Mock()
        bad_response.status_code = 200
        bad_response.headers = {"Content-Type": "application/json"}
        bad_response.text = ""
        bad_response.json = Mock(return_value={})

        with patch("app.services.openalex_client.requests.get", return_value=bad_response):
            papers = fetch_openalex_papers("query", max_results=1)

        self.assertEqual(papers, [])


if __name__ == "__main__":
    unittest.main()
