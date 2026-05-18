from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.core.state import ResearchState
from app.services.llm_service import generate_answer


class ReasoningAgent(BaseAgent):
    """Generate the final answer from retrieved chunks."""

    name = "ReasoningAgent"

    def run(self, state: ResearchState) -> ResearchState:
        """Generate the final answer and populate state.generated_answer."""
        try:
            if not state.retrieved_chunks:
                if not state.generated_answer:
                    state.generated_answer = "No relevant documents found for the query."
                self._log("No retrieved chunks available for reasoning.")
                return state

            answer = generate_answer(
                state.query,
                state.evidence_objects,
                state.filtered_papers,
                state.metadata_store,
                state.papers_with_extracted_text,
                query_intent=state.query_intent,
                diagnostics=state.diagnostics,
            )
            state.generated_answer = answer
            state.references = [
                state.metadata_store[paper_id]
                for paper_id in sorted(state.papers_with_extracted_text)
                if paper_id in state.metadata_store
            ]
            self._log(f"Generated answer with {len(state.references)} references")
        except Exception as error:
            error_message = f"Reasoning failed: {error}"
            self._log(error_message)
            state.errors.append(error_message)
        return state
