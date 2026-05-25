from app.agents.acquisition_agent import AcquisitionAgent
from app.agents.compression_agent import CompressionAgent
from app.agents.evidence_extractor_agent import EvidenceExtractorAgent
from app.agents.query_agent import QueryAgent
from app.agents.query_expansion_agent import QueryExpansionAgent
from app.agents.relevance_verifier_agent import RelevanceVerifierAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.retriever_agent import RetrieverAgent
from app.core.state import ResearchState
from typing import Any, Callable, Dict, Optional


def run_pipeline_state(
    query: str,
    progress_callback: Optional[Callable[[str, str, str, Optional[int], Optional[ResearchState]], None]] = None,
) -> ResearchState:
    """Run the research pipeline and return the final state for API consumption."""
    state = ResearchState(query=query)
    agents = [
        QueryAgent(),
        QueryExpansionAgent(),
        RetrievalAgent(),
        RelevanceVerifierAgent(),
        AcquisitionAgent(),
        RetrieverAgent(),
        CompressionAgent(),
        EvidenceExtractorAgent(),
        ReasoningAgent(),
    ]

    total_agents = len(agents)
    for index, agent in enumerate(agents):
        state = agent.execute(state, callback=progress_callback)
        if progress_callback:
            progress = round(((index + 1) / total_agents) * 100)
            progress_callback(agent.name, "progress", f"{agent.name} finished", progress, state)

    return state


def run_pipeline(query: str) -> str:
    """Run the research pipeline using lightweight agent orchestration."""
    return run_pipeline_state(query).generated_answer


def serialize_state(state: ResearchState) -> Dict[str, Any]:
    return {
        "query": state.query,
        "query_intent": state.query_intent,
        "papers": state.papers,
        "generated_answer": state.generated_answer,
        "references": state.references,
        "diagnostics": state.diagnostics,
        "errors": state.errors,
        "expanded_queries": state.expanded_queries,
        "filtered_papers": state.filtered_papers,
        "evidence_objects": state.evidence_objects,
        "papers_with_extracted_text": list(state.papers_with_extracted_text),
    }
