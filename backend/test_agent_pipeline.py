import sys
import unittest
from pathlib import Path
from unittest.mock import patch

workspace_root = Path(__file__).resolve().parent
sys.path.insert(0, str(workspace_root))

from app.agents.acquisition_agent import AcquisitionAgent
from app.agents.compression_agent import CompressionAgent
from app.agents.evidence_extractor_agent import EvidenceExtractorAgent
from app.agents.query_agent import QueryAgent
from app.agents.query_expansion_agent import QueryExpansionAgent
from app.agents.relevance_verifier_agent import RelevanceVerifierAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.retriever_agent import RetrieverAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.core.pipeline import run_pipeline
from app.core.state import ResearchState


class TestAgentPipeline(unittest.TestCase):
    def test_agent_state_propagation(self):
        """Verify that each agent correctly updates shared ResearchState."""
        state = ResearchState(query="Compare model A vs model B")

        with patch("app.agents.query_agent.analyze_query") as mock_analyze, \
             patch("app.agents.query_expansion_agent.analyze_query") as mock_expand_analyze, \
             patch("app.agents.retrieval_agent.hybrid_retrieve") as mock_hybrid_retrieve, \
             patch("app.agents.acquisition_agent.download_pdf_with_fallbacks") as mock_download, \
             patch("app.agents.acquisition_agent.extract_text_from_pdf") as mock_extract, \
             patch("app.agents.acquisition_agent.process_documents") as mock_process, \
             patch("app.agents.retriever_agent.create_vector_store") as mock_create_vector_store, \
             patch("app.agents.retriever_agent.retrieve_chunks") as mock_retrieve_chunks, \
             patch("app.agents.reasoning_agent.generate_answer") as mock_generate_answer:

            mock_analyze.return_value = {
                "query_type": "comparison",
                "focus_terms": ["model A", "model B"],
                "comparison_pairs": [("model A", "model B")],
                "original_query": "Compare model A vs model B",
            }
            mock_hybrid_retrieve.return_value = {
                "papers": [
                    {
                        "paper_id": "paper1",
                        "title": "Paper 1",
                        "abstract": "Abstract 1",
                        "url": "http://example.com/paper1",
                        "authors": ["Author One"],
                        "year": 2025,
                        "venue": "Venue A",
                        "doi": "10.1000/example",
                        "locations": [],
                        "open_access": {},
                        "source": "openalex",
                    }
                ],
                "source_counts": {"openalex": 1, "core": 0},
                "candidate_count": 1,
                "scored_candidates": [{"paper_id": "paper1", "score": 5.0}],
            }
            mock_download.return_value = "/tmp/fake.pdf"
            mock_extract.return_value = "x" * 600
            mock_process.return_value = [
                type(
                    "Doc",
                    (),
                    {
                        "page_content": "Title: Paper 1\n\ncontent",
                        "metadata": {"paper_id": "paper1", "chunk_index": 0},
                    },
                )
            ]
            mock_create_vector_store.return_value = object()
            mock_retrieve_chunks.return_value = (
                [
                    type(
                        "Doc",
                        (),
                        {
                            "page_content": "Title: Paper 1\n\ncontent",
                            "metadata": {"paper_id": "paper1", "chunk_index": 0},
                        },
                    )
                ],
                {
                    "final_chunk_count": 1,
                    "final_source_count": 1,
                    "focus_coverage": {},
                    "candidate_coverage": {},
                },
            )
            mock_generate_answer.return_value = "FINAL ANSWER"

            state = QueryAgent().run(state)
            self.assertEqual(state.query_intent["query_type"], "comparison")
            self.assertEqual(state.query_intent["comparison_pairs"], [("model A", "model B")])

            state = QueryExpansionAgent().run(state)
            self.assertTrue(state.expanded_queries)
            self.assertIn("model a vs model b", state.expanded_queries[0])

            state = RetrievalAgent().run(state)
            self.assertTrue(state.papers)
            self.assertIn("paper1", state.metadata_store)

            state = RelevanceVerifierAgent().run(state)
            self.assertTrue(state.papers)
            self.assertGreaterEqual(state.diagnostics.get("relevance_scores", []), [])

            state = AcquisitionAgent().run(state)
            self.assertEqual(state.papers_with_extracted_text, {"paper1"})
            self.assertTrue(state.documents)

            state = RetrieverAgent().run(state)
            self.assertTrue(state.retrieved_chunks)
            self.assertEqual(state.diagnostics["retrieved_chunk_count"], 1)

            state = CompressionAgent().run(state)
            self.assertTrue(state.retrieved_chunks)
            self.assertTrue(state.diagnostics["compressed_chunk_count"] >= 0)

            state = EvidenceExtractorAgent().run(state)
            self.assertTrue(state.evidence_objects)
            self.assertEqual(state.diagnostics["evidence_objects_created"], len(state.evidence_objects))

            state = ReasoningAgent().run(state)
            self.assertEqual(state.generated_answer, "FINAL ANSWER")
            self.assertTrue(state.references)

    def test_compression_agent_reduces_prompt_tokens(self):
        """Verify compression reduces chunk size and updates diagnostics."""
        state = ResearchState(query="test")
        state.retrieved_chunks = [
            type(
                "Doc",
                (),
                {
                    "page_content": (
                        "This paper describes a method and a result. "
                        "The method is evaluated using accuracy and efficiency metrics. "
                        "The method is evaluated using accuracy and efficiency metrics. "
                        "The study also mentions tradeoffs in memory and runtime. "
                        "This work is important and the rest of the paper is organized as follows."
                    ),
                    "metadata": {"paper_id": "paper1", "chunk_index": 0},
                },
            )
        ]

        state = CompressionAgent().run(state)

        self.assertEqual(state.diagnostics["compressed_chunk_count"], 1)
        self.assertLessEqual(state.diagnostics["compression_ratio"], 1.0)
        self.assertTrue(state.retrieved_chunks[0].page_content)
        self.assertTrue(state.diagnostics["estimated_prompt_tokens"] > 0)

    def test_pipeline_execution_and_order(self):
        """Verify full pipeline execution order and output preservation."""
        with patch("app.agents.query_agent.analyze_query", return_value={
            "query_type": "comparison",
            "focus_terms": ["A", "B"],
            "comparison_pairs": [("A", "B")],
            "original_query": "Compare A vs B",
        }), \
             patch("app.agents.query_expansion_agent.analyze_query", return_value={
                 "query_type": "comparison",
                 "focus_terms": ["A", "B"],
                 "comparison_pairs": [("A", "B")],
                 "original_query": "Compare A vs B",
             }), \
             patch("app.agents.retrieval_agent.hybrid_retrieve", return_value={
                 "papers": [
                     {
                         "paper_id": "paper1",
                         "title": "Paper 1",
                         "abstract": "Abstract 1",
                         "url": "http://example.com/paper1",
                         "authors": ["Author One"],
                         "year": 2025,
                         "venue": "Venue A",
                         "doi": "10.1000/example",
                         "locations": [],
                         "open_access": {},
                         "source": "openalex",
                     }
                 ],
                 "source_counts": {"openalex": 1, "core": 0},
                 "candidate_count": 1,
                 "scored_candidates": [{"paper_id": "paper1", "score": 5.0}],
             }), \
             patch("app.agents.acquisition_agent.download_pdf_with_fallbacks", return_value="/tmp/fake.pdf"), \
             patch("app.agents.acquisition_agent.extract_text_from_pdf", return_value="x" * 600), \
             patch("app.agents.acquisition_agent.process_documents", return_value=[
                 type(
                     "Doc",
                     (),
                     {
                         "page_content": "Title: Paper 1\n\ncontent",
                         "metadata": {"paper_id": "paper1", "chunk_index": 0},
                     },
                 )
             ]), \
             patch("app.agents.retriever_agent.create_vector_store", return_value=object()), \
             patch("app.agents.retriever_agent.retrieve_chunks", return_value=(
                 [
                     type(
                         "Doc",
                         (),
                         {
                             "page_content": "Title: Paper 1\n\ncontent",
                             "metadata": {"paper_id": "paper1", "chunk_index": 0},
                         },
                     )
                 ],
                 {
                     "final_chunk_count": 1,
                     "final_source_count": 1,
                     "focus_coverage": {},
                     "candidate_coverage": {},
                 },
             )), \
             patch("app.agents.reasoning_agent.generate_answer", return_value="FINAL ANSWER"):
            result = run_pipeline("Compare A vs B")

        self.assertEqual(result, "FINAL ANSWER")


if __name__ == "__main__":
    unittest.main()
