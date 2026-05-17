from __future__ import annotations

from collections import defaultdict
from typing import Any, List

from app.agents.base_agent import BaseAgent
from app.core.state import ResearchState
from app.services.vector_store import create_vector_store
from app.services.retriever import retrieve_chunks


class RetrieverAgent(BaseAgent):
    """Build a vector store, retrieve and rerank chunks, and update state."""

    name = "RetrieverAgent"

    def run(self, state: ResearchState) -> ResearchState:
        """Filter, index, and retrieve chunks from the processed documents."""
        try:
            if not state.documents:
                self._log("No processed documents available for retrieval.")
                return state

            filtered_chunks = [
                chunk for chunk in state.documents
                if chunk.metadata.get("paper_id") in state.papers_with_extracted_text
            ]
            self._log(f"Filtered {len(filtered_chunks)} chunks based on extracted text")

            if not filtered_chunks:
                state.generated_answer = "No relevant documents found for the query."
                return state

            chunks_by_paper = defaultdict(list)
            for chunk in filtered_chunks:
                paper_id = chunk.metadata.get("paper_id")
                if paper_id:
                    chunks_by_paper[paper_id].append(chunk)

            capped_chunks: List[Any] = []
            for paper_id, paper_chunks in chunks_by_paper.items():
                capped_chunks.extend(paper_chunks[:300])

            self._log(f"Capped chunks to {len(capped_chunks)} total across papers")
            if not capped_chunks:
                state.generated_answer = "No relevant documents found for the query."
                return state

            vector_store = create_vector_store(capped_chunks)
            self._log("Vector store created")

            retrieved, retrieval_metrics = retrieve_chunks(vector_store, state.query, query_intent=state.query_intent)
            state.retrieved_chunks = retrieved
            state.diagnostics["retrieved_chunk_count"] = retrieval_metrics.get("final_chunk_count", len(retrieved))
            state.diagnostics["candidate_chunk_count"] = retrieval_metrics.get("candidate_chunk_count", 0)
            state.diagnostics["source_count"] = retrieval_metrics.get("final_source_count", 0)
            state.diagnostics["focus_coverage"] = retrieval_metrics.get("focus_coverage", {})
            state.diagnostics["candidate_coverage"] = retrieval_metrics.get("candidate_coverage", {})
            self._log(f"Retrieved {len(retrieved)} chunks from {state.diagnostics['source_count']} sources")
        except Exception as error:
            error_message = f"Retriever failed: {error}"
            self._log(error_message)
            state.errors.append(error_message)
        return state
