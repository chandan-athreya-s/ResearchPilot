from __future__ import annotations

from typing import Any

from app.agents.base_agent import BaseAgent
from app.core.state import ResearchState
from app.services.hybrid_retrieval import hybrid_retrieve


class RetrievalAgent(BaseAgent):
    """Retrieve papers using hybrid OpenAlex + CORE retrieval and create a metadata snapshot."""

    name = "RetrievalAgent"

    def run(self, state: ResearchState) -> ResearchState:
        """Fetch papers for the query and update state.papers and state.metadata_store."""
        try:
            hybrid_results = hybrid_retrieve(
                query=state.query,
                expanded_queries=state.expanded_queries or [state.query],
                query_intent=state.query_intent,
                max_results_per_source=7,
            )
            papers = hybrid_results.get("papers", [])
            state.papers = papers
            state.metadata_store = {paper["paper_id"]: paper.copy() for paper in papers}
            state.diagnostics["papers_retrieved"] = len(papers)
            state.diagnostics["paper_count"] = len(papers)
            state.diagnostics["retrieval_source_counts"] = hybrid_results.get("source_counts", {})
            state.diagnostics["candidate_count"] = hybrid_results.get("candidate_count", 0)
            self._log(f"Fetched {len(papers)} papers from hybrid retrieval")
        except Exception as error:
            error_message = f"Paper retrieval failed: {error}"
            self._log(error_message)
            state.errors.append(error_message)
        return state
