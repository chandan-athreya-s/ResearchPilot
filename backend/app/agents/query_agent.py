from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.core.state import ResearchState
from app.services.query_analyzer import analyze_query


class QueryAgent(BaseAgent):
    """Analyze the incoming query and populate intent metadata."""

    name = "QueryAgent"

    def run(self, state: ResearchState) -> ResearchState:
        """Analyze the query and update state.query_intent."""
        try:
            state.query_intent = analyze_query(state.query)
            state.diagnostics["query_type"] = state.query_intent.get("query_type", "")
            self._log(f"Detected query type: {state.query_intent.get('query_type')}")
            self._log(f"Focus terms: {', '.join(state.query_intent.get('focus_terms', []))}")
        except Exception as error:
            error_message = f"Query analysis failed: {error}"
            self._log(error_message)
            state.errors.append(error_message)
        return state
