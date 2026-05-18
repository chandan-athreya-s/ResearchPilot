import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

workspace_root = Path(__file__).resolve().parent
sys.path.insert(0, str(workspace_root))

from app.agents.evidence_extractor_agent import EvidenceExtractorAgent
from app.core.evidence import EvidenceObject, build_evidence_object, extract_metrics, extract_findings, extract_tradeoffs
from app.core.state import ResearchState


class TestEvidenceExtraction(unittest.TestCase):
    def test_build_evidence_object_contains_structured_fields(self):
        chunk = type(
            "Doc",
            (),
            {
                "page_content": (
                    "Method: We use a transformer-based retrieval model. "
                    "Evaluation on ImageNet and COCO shows 92.4% accuracy and 8.7ms latency. "
                    "The approach outperforms the baseline while incurring higher memory overhead. "
                    "Limitations include sensitivity to noisy labels and higher training cost."
                ),
                "metadata": {"paper_id": "p1", "chunk_index": 0},
            },
        )
        metadata = {
            "paper_id": "p1",
            "title": "Demo Paper",
            "authors": ["Alice Smith"],
            "year": 2026,
            "venue": "TestConf",
            "doi": "10.1234/test",
        }

        evidence = build_evidence_object(chunk, metadata, {"focus_terms": ["retrieval model"], "query_type": "comparison"})

        self.assertIsInstance(evidence, EvidenceObject)
        self.assertIn("transformer", evidence.method.lower())
        self.assertTrue(any("accuracy" in metric.lower() or "latency" in metric.lower() for metric in evidence.metrics))
        self.assertTrue(any("outperforms" in finding.lower() or "outperform" in finding.lower() for finding in evidence.findings))
        self.assertTrue(any("higher memory" in tradeoff.lower() or "trade" in tradeoff.lower() for tradeoff in evidence.tradeoffs))
        self.assertTrue(any("sensitivity" in limitation.lower() for limitation in evidence.limitations))
        self.assertIn("retrieval model", evidence.relevance_to_query.lower())

    def test_extract_metrics_and_findings_and_tradeoffs(self):
        text = (
            "In experiments on CIFAR and ImageNet, the method achieved 95.2% accuracy and 0.12s runtime. "
            "It outperforms prior baselines while trading off memory for throughput. "
            "A limitation is higher inference cost."
        )
        metrics = extract_metrics(text)
        findings = extract_findings(text)
        tradeoffs = extract_tradeoffs(text)

        self.assertTrue(any("95.2%" in m or "0.12s" in m for m in metrics))
        self.assertTrue(any("outperforms" in f for f in findings))
        self.assertTrue(any("trading off memory" in t or "trade off" in t.lower() for t in tradeoffs))

    def test_evidence_extractor_agent_populates_state_and_diagnostics(self):
        state = ResearchState(query="Compare A vs B")
        state.query_intent = {"query_type": "comparison", "focus_terms": ["A", "B"]}
        state.metadata_store = {
            "p1": {"paper_id": "p1", "title": "Paper One", "authors": ["Author One"], "year": 2025},
            "p2": {"paper_id": "p2", "title": "Paper Two", "authors": ["Author Two"], "year": 2024},
        }
        state.retrieved_chunks = [
            type(
                "Doc",
                (),
                {
                    "page_content": "Approach A uses a transformer. Results show 88% accuracy. Tradeoff is memory.",
                    "metadata": {"paper_id": "p1", "chunk_index": 0},
                },
            ),
            type(
                "Doc",
                (),
                {
                    "page_content": "Approach B uses a retrieval augmented system. Results show 85% accuracy but lower latency.",
                    "metadata": {"paper_id": "p2", "chunk_index": 1},
                },
            ),
        ]

        state = EvidenceExtractorAgent().run(state)

        self.assertEqual(len(state.evidence_objects), 2)
        self.assertEqual(state.diagnostics["evidence_objects_created"], 2)
        self.assertGreaterEqual(state.diagnostics["extracted_metrics_count"], 2)
        self.assertGreaterEqual(state.diagnostics["extracted_findings_count"], 2)
        self.assertGreaterEqual(state.diagnostics["extracted_tradeoffs_count"], 1)
        self.assertEqual(state.diagnostics["evidence_diversity"]["unique_papers"], 2)

    def test_generate_answer_uses_structured_evidence(self):
        from app.services.llm_service import generate_answer

        evidence_objects = [
            EvidenceObject(
                paper_id="p1",
                paper_title="Paper One",
                authors=["Author One"],
                year=2025,
                method="Transformer-based retrieval",
                dataset="ImageNet",
                benchmark="ImageNet benchmark",
                metrics=["95.2% accuracy"],
                findings=["Outperforms baseline"],
                limitations=["Higher memory use"],
                tradeoffs=["Memory vs throughput"],
                relevance_to_query="Matches focus terms: retrieval",
                source_reference={"paper_id": "p1", "title": "Paper One"},
                extracted_text="Transformer-based retrieval model with strong accuracy.",
            )
        ]

        state = ResearchState(query="Compare retrieval methods")
        state.query_intent = {"query_type": "comparison", "focus_terms": ["retrieval"]}
        state.metadata_store = {"p1": {"paper_id": "p1", "title": "Paper One", "authors": ["Author One"], "year": 2025}}
        state.papers_with_extracted_text = {"p1"}

        mock_client = Mock()
        mock_client.generate.return_value = Mock(generations=[[Mock(text="Final answer [1]")]])

        with patch("app.services.llm_service._get_ollama_client", return_value=mock_client):
            answer = generate_answer(
                state.query,
                evidence_objects,
                state.papers,
                state.metadata_store,
                state.papers_with_extracted_text,
                state.query_intent,
                state.diagnostics,
            )

        self.assertIn("Final answer", answer)


if __name__ == "__main__":
    unittest.main()
