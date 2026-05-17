from app.agents.acquisition_agent import AcquisitionAgent
from app.agents.compression_agent import CompressionAgent
from app.agents.query_agent import QueryAgent
from app.agents.query_expansion_agent import QueryExpansionAgent
from app.agents.relevance_verifier_agent import RelevanceVerifierAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.retriever_agent import RetrieverAgent
from app.core.state import ResearchState


def run_pipeline(query: str) -> str:
    """Run the research pipeline using lightweight agent orchestration."""
    state = ResearchState(query=query)
    agents = [
        QueryAgent(),
        QueryExpansionAgent(),
        RetrievalAgent(),
        RelevanceVerifierAgent(),
        AcquisitionAgent(),
        RetrieverAgent(),
        CompressionAgent(),
        ReasoningAgent(),
    ]

    for agent in agents:
        state = agent.execute(state)

    return state.generated_answer
