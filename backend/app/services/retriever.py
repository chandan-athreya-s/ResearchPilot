from sentence_transformers import CrossEncoder
from collections import defaultdict
from typing import Dict, List, Optional

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def retrieve_chunks(vector_store, query, query_intent: Optional[Dict] = None):
    """Retrieve relevant chunks with query-aware adjustments.
    
    Args:
        vector_store: LangChain vector store (FAISS-backed)
        query: The research query string
        query_intent: Optional dict with keys:
            - query_type: str (comparison, survey, implementation, challenges, general)
            - focus_terms: List[str] (important terms in the query)
    """
    # Determine retrieval parameters based on query type
    if query_intent is None:
        query_intent = {"query_type": "general", "focus_terms": []}
    
    query_type = query_intent.get("query_type", "general")
    focus_terms = query_intent.get("focus_terms", [])
    
    # Adjust k based on query type
    if query_type == "survey":
        k = 40  # Retrieve more for diversity
    elif query_type == "comparison":
        k = 35  # Retrieve more to balance across topics
    else:
        k = 30  # Default
    
    # Step 1 — retrieve a large candidate pool from FAISS:
    candidate_docs = vector_store.similarity_search(query, k=k)
    
    # Step 2 — optionally filter for focus terms (for challenges queries)
    if query_type == "challenges" and focus_terms:
        candidate_docs = _prioritize_for_focus_terms(candidate_docs, focus_terms, priority="limitations")
    
    # Step 3 — group candidates by source paper ID:
    grouped = defaultdict(list)
    for doc in candidate_docs:
        source_id = doc.metadata.get("paper_id", "unknown")
        grouped[source_id].append(doc)

    # Step 4 — adjust chunks per source based on query type:
    chunks_per_source = _get_chunks_per_source(query_type)
    capped = []
    for source_id, docs in grouped.items():
        capped.extend(docs[:chunks_per_source])

    # Step 5 — rerank the capped pool:
    reranked = rerank(query, capped)
    
    # Step 6 — take top N based on query type:
    final_k = _get_final_k(query_type)
    final_docs = reranked[:final_k]

    # Step 7 — log which sources made it through:
    source_ids = set(d.metadata.get("paper_id", "unknown") for d in final_docs)
    print(f"✓ Query type: {query_type} | Final chunks drawn from {len(source_ids)} sources: {source_ids}")
    if focus_terms:
        print(f"  Focus terms: {', '.join(focus_terms)}")

    return final_docs


def _get_chunks_per_source(query_type: str) -> int:
    """Determine how many chunks to keep per source based on query type."""
    if query_type == "survey":
        return 2  # Maximize source diversity
    elif query_type == "comparison":
        return 3  # Balanced across topics
    else:
        return 3  # Default


def _get_final_k(query_type: str) -> int:
    """Determine final number of chunks to return based on query type."""
    if query_type == "survey":
        return 7  # More diverse chunks
    elif query_type == "comparison":
        return 6  # Extra chunks for comparing approaches
    else:
        return 5  # Default


def _prioritize_for_focus_terms(docs: List, focus_terms: List[str], priority: str = "general") -> List:
    """Prioritize documents based on focus terms or content type.
    
    Args:
        docs: List of document chunks
        focus_terms: Important terms from the query
        priority: Type of content to prioritize (general, limitations, technical)
    
    Returns:
        Reordered list with prioritized docs first
    """
    if not focus_terms and priority == "general":
        return docs
    
    # Keywords for different content types
    limitation_keywords = {
        "limitation", "challenge", "problem", "issue", "difficulty",
        "constraint", "limitation", "weakness", "gap", "drawback",
        "fail", "failure", "error", "incomplete", "scalability",
        "performance", "trade-off", "trade-offs", "overhead"
    }
    
    technical_keywords = {
        "algorithm", "architecture", "framework", "method", "technique",
        "approach", "implementation", "system", "module", "component",
        "optimization", "optimization", "efficiency", "complexity"
    }
    
    def score_doc(doc, terms, keywords):
        """Score a document based on presence of focus terms and keywords."""
        content = doc.page_content.lower()
        score = 0
        
        # Higher score for focus terms
        for term in terms:
            if term.lower() in content:
                score += 3
        
        # Lower score for keyword presence
        for keyword in keywords:
            if keyword in content:
                score += 1
        
        return score
    
    # Score based on priority type
    if priority == "limitations":
        scored_docs = [(doc, score_doc(doc, focus_terms, limitation_keywords)) for doc in docs]
    elif priority == "technical":
        scored_docs = [(doc, score_doc(doc, focus_terms, technical_keywords)) for doc in docs]
    else:
        scored_docs = [(doc, score_doc(doc, focus_terms, set())) for doc in docs]
    
    # Sort by score descending, maintaining relative order for ties
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    return [doc for doc, _ in scored_docs]

def rerank(query, docs):
    if not docs:
        return []
    passages = [doc.page_content for doc in docs]
    rerank_scores = reranker.predict([(query, passage) for passage in passages])
    reranked = sorted(zip(docs, rerank_scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked]