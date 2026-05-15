from sentence_transformers import CrossEncoder
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple, Set
import re

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def retrieve_for_focus(vector_store, focus_term: str, k: int = 15) -> List:
    """Retrieve chunks relevant to a specific focus term.
    
    Args:
        vector_store: LangChain vector store
        focus_term: Single focus term or phrase
        k: Number of results to retrieve
    
    Returns:
        List of relevant chunks for this focus term
    """
    if not focus_term or len(focus_term) < 2:
        return []
    
    try:
        results = vector_store.similarity_search(focus_term, k=k)
        return results
    except Exception as e:
        print(f"⚠ Focus retrieval failed for '{focus_term}': {e}")
        return []


def compute_focus_coverage(docs: List, focus_terms: List[str]) -> Dict[str, int]:
    """Compute how many chunks cover each focus term.
    
    Args:
        docs: List of document chunks
        focus_terms: List of focus terms to check coverage for
    
    Returns:
        Dict mapping focus_term → count of chunks mentioning it
    """
    coverage = {}
    normalized_docs = [_normalize_text(doc.page_content) for doc in docs]
    
    for term in focus_terms:
        normalized_term = _normalize_text(term)
        count = 0
        for content in normalized_docs:
            if normalized_term in content:
                count += 1
        coverage[term] = count
    
    return coverage


def merge_and_deduplicate_focused(focus_retrievals: Dict[str, List], max_per_focus: int = 5) -> List:
    """Merge retrieved chunks from multiple focus terms with deduplication.
    
    Args:
        focus_retrievals: Dict mapping focus_term → list of chunks
        max_per_focus: Maximum chunks to keep per focus
    
    Returns:
        Merged list of deduplicated chunks, ordered by frequency and relevance
    """
    if not focus_retrievals:
        return []
    
    merged = []
    seen_hashes = set()
    focus_counts = defaultdict(int)
    
    # First pass: add chunks while limiting per-focus and removing duplicates
    for focus_term, chunks in focus_retrievals.items():
        for chunk in chunks[:max_per_focus]:
            # Compute content hash for deduplication
            content_hash = hash(_normalize_text(chunk.page_content))
            
            if content_hash not in seen_hashes:
                # Add focus term tracking to metadata
                if "focus_hits" not in chunk.metadata:
                    chunk.metadata["focus_hits"] = []
                chunk.metadata["focus_hits"].append(focus_term)
                
                merged.append(chunk)
                seen_hashes.add(content_hash)
                focus_counts[focus_term] += 1
            else:
                # Chunk already added from another focus - update focus hits
                for existing in merged:
                    if hash(_normalize_text(existing.page_content)) == content_hash:
                        if "focus_hits" not in existing.metadata:
                            existing.metadata["focus_hits"] = []
                        if focus_term not in existing.metadata["focus_hits"]:
                            existing.metadata["focus_hits"].append(focus_term)
                        break
    
    print(f"  Focus-aware merging: Merged {len(merged)} deduplicated chunks across {len(focus_retrievals)} focus areas")
    return merged


def boost_underrepresented_focus(vector_store, merged_docs: List, focus_terms: List[str], 
                                 coverage_threshold: int = 2, additional_k: int = 5) -> List:
    """Add supplemental retrieval for focus terms with weak coverage.
    
    Args:
        vector_store: LangChain vector store
        merged_docs: Already merged documents
        focus_terms: List of focus terms
        coverage_threshold: Minimum chunks required per focus
        additional_k: How many additional chunks to retrieve per weak focus
    
    Returns:
        Merged docs with boosted underrepresented focuses
    """
    if not focus_terms or not merged_docs:
        return merged_docs
    
    # Compute current coverage
    current_coverage = compute_focus_coverage(merged_docs, focus_terms)
    seen_hashes = {hash(_normalize_text(doc.page_content)) for doc in merged_docs}
    
    # Identify weak focus areas
    weak_focuses = [term for term in focus_terms if current_coverage.get(term, 0) < coverage_threshold]
    
    if not weak_focuses:
        return merged_docs
    
    print(f"  Weak focus areas detected: {weak_focuses}")
    
    # Retrieve supplemental chunks for weak focuses
    supplemental = []
    for weak_term in weak_focuses:
        focus_results = retrieve_for_focus(vector_store, weak_term, k=additional_k)
        
        for doc in focus_results:
            content_hash = hash(_normalize_text(doc.page_content))
            if content_hash not in seen_hashes:
                if "focus_hits" not in doc.metadata:
                    doc.metadata["focus_hits"] = []
                doc.metadata["focus_hits"].append(weak_term)
                doc.metadata["boost_reason"] = f"weak_coverage_{weak_term}"
                
                supplemental.append(doc)
                seen_hashes.add(content_hash)
    
    if supplemental:
        print(f"  Weak focus boost: Added {len(supplemental)} chunks for {len(weak_focuses)} weak areas")
        merged_docs.extend(supplemental)
    
    return merged_docs


def rerank_with_focus_awareness(query: str, docs: List, focus_terms: List[str] = None) -> List:
    """Rerank documents considering both relevance and focus coverage.
    
    Args:
        query: The original query
        docs: Chunks to rerank
        focus_terms: Focus terms to weight in reranking
    
    Returns:
        Reranked chunks with relevance + focus balancing
    """
    if not docs:
        return []
    
    # Get base relevance scores from CrossEncoder
    passages = [doc.page_content for doc in docs]
    relevance_scores = reranker.predict([(query, passage) for passage in passages])
    
    # Add focus-aware adjustments if focus terms provided
    adjusted_scores = list(relevance_scores)
    
    if focus_terms:
        for i, doc in enumerate(docs):
            # Boost if this doc hits multiple focus terms
            focus_hits = doc.metadata.get("focus_hits", [])
            if focus_hits:
                # Small boost for each additional focus hit (prevents over-balancing)
                focus_boost = len(focus_hits) * 0.1
                adjusted_scores[i] = adjusted_scores[i] + focus_boost
            
            # Slight penalty for docs that miss all focus terms
            content_lower = _normalize_text(doc.page_content)
            if not any(_normalize_text(term) in content_lower for term in focus_terms):
                adjusted_scores[i] = adjusted_scores[i] - 0.05
    
    # Rerank by adjusted scores
    reranked = sorted(zip(docs, adjusted_scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked]

def retrieve_chunks(vector_store, query, query_intent: Optional[Dict] = None):
    """Retrieve relevant chunks with query-aware adjustments.
    
    Args:
        vector_store: LangChain vector store (FAISS-backed)
        query: The research query string
        query_intent: Optional dict with keys:
            - query_type: str (comparison, survey, implementation, challenges, general)
            - focus_terms: List[str] (important terms in the query)
    """
    if query_intent is None:
        query_intent = {"query_type": "general", "focus_terms": []}
    
    query_type = query_intent.get("query_type", "general")
    focus_terms = query_intent.get("focus_terms", [])
    
    # Adjust k based on query type
    if query_type == "survey":
        k = 50  # Retrieve more candidates for broader coverage
    elif query_type == "comparison":
        k = 45  # Retrieve more candidates for both comparison sides
    else:
        k = 35  # Default
    
    # Step 1 — retrieve a broad candidate pool from FAISS:
    base_docs = vector_store.similarity_search(query, k=k)

    # Step 2 — build separate focus retrievals for survey/comparison queries
    if query_type in {"comparison", "survey"} and focus_terms:
        focus_retrievals = {
            term: retrieve_for_focus(vector_store, term, k=min(15, k))
            for term in focus_terms
        }
        focus_docs = merge_and_deduplicate_focused(focus_retrievals, max_per_focus=4)
        focus_docs = boost_underrepresented_focus(vector_store, focus_docs, focus_terms, coverage_threshold=2, additional_k=5)

        # Keep the focused docs first, but preserve relevance by appending base retrieval results
        candidate_docs = []
        seen_hashes = set()
        for doc in focus_docs:
            content_hash = hash(_normalize_text(doc.page_content))
            if content_hash not in seen_hashes:
                candidate_docs.append(doc)
                seen_hashes.add(content_hash)
        for doc in base_docs:
            content_hash = hash(_normalize_text(doc.page_content))
            if content_hash not in seen_hashes:
                candidate_docs.append(doc)
                seen_hashes.add(content_hash)
    else:
        candidate_docs = base_docs

    # Step 3 — optionally prioritize challenge-related content
    if query_type == "challenges" and focus_terms:
        candidate_docs = _prioritize_for_focus_terms(candidate_docs, focus_terms, priority="limitations")

    # Step 4 — rerank candidates with focus awareness for coverage and relevance
    if query_type in {"comparison", "survey"} and focus_terms:
        candidate_docs = rerank_with_focus_awareness(query, candidate_docs, focus_terms)
    else:
        candidate_docs = rerank(query, candidate_docs)

    # Step 5 — group candidates by source paper ID:
    grouped = defaultdict(list)
    for doc in candidate_docs:
        source_id = doc.metadata.get("paper_id", "unknown")
        grouped[source_id].append(doc)

    # Step 6 — adjust chunks per source based on query type and interleave by paper source
    chunks_per_source = _get_chunks_per_source(query_type)
    capped = _interleave_by_source(grouped, chunks_per_source)

    # Step 7 — rerank the capped pool:
    reranked = rerank(query, capped)
    
    # Step 8 — choose a final balanced set with source/topic coverage
    final_k = _get_final_k(query_type)
    final_docs, diagnostics = _choose_diverse_docs(
        reranked,
        final_k,
        max_per_source=chunks_per_source,
        query_type=query_type,
        query=query,
    )

    # Step 7 — log diagnostics and coverage information
    candidate_sources = set(d.metadata.get("paper_id", "unknown") for d in candidate_docs)
    source_ids = set(d.metadata.get("paper_id", "unknown") for d in final_docs)
    coverage_terms = list(focus_terms)
    if query_type == "comparison":
        coverage_terms.extend(_extract_comparison_sides(query))
    coverage_terms = [term for term in coverage_terms if term]
    coverage = _compute_term_coverage(final_docs, coverage_terms)
    candidate_coverage = _compute_term_coverage(candidate_docs, coverage_terms)

    print(f"✓ Query type: {query_type} | Candidate chunks: {len(candidate_docs)} | Candidate sources: {len(candidate_sources)}")
    print(f"  Final chunks: {len(final_docs)} | Unique sources: {len(source_ids)} | Sources: {source_ids}")
    if focus_terms:
        print(f"  Focus terms: {', '.join(focus_terms)}")
    if coverage_terms:
        print(f"  Coverage: {coverage} | Candidate coverage: {candidate_coverage}")
    if diagnostics:
        reason_counts = defaultdict(int)
        for reason, _ in diagnostics:
            reason_counts[reason] += 1
        print(f"  Filtering diagnostics: {dict(reason_counts)}")

    return final_docs


def _get_chunks_per_source(query_type: str) -> int:
    """Determine how many chunks to keep per source based on query type."""
    if query_type == "survey":
        return 4  # Allow multiple useful chunks from the same source
    elif query_type == "comparison":
        return 5  # Allow strong papers to contribute broader evidence
    else:
        return 5  # Default


def _get_final_k(query_type: str) -> int:
    """Determine final number of chunks to return based on query type."""
    if query_type == "survey":
        return 9  # More coverage for broad queries
    elif query_type == "comparison":
        return 9  # Better coverage of both comparison sides
    else:
        return 7  # Default


def _interleave_by_source(grouped: Dict[str, List], chunks_per_source: int) -> List:
    """Interleave candidate chunks across papers to prevent one source from dominating early slots."""
    interleaved = []
    paper_ids = list(grouped.keys())
    for i in range(chunks_per_source):
        for paper_id in paper_ids:
            docs = grouped.get(paper_id, [])
            if i < len(docs):
                interleaved.append(docs[i])
    return interleaved


def _choose_diverse_docs(docs: List, final_k: int, max_per_source: int, query_type: str, query: str = "") -> Tuple[List, List[tuple]]:
    """Select the final chunk set with source/topic diversity while preserving relevance."""
    selected = []
    source_counts = defaultdict(int)
    diagnostics = []

    # 1. Enforce minimal comparison coverage first, if applicable
    if query_type == "comparison":
        comparison_sides = _extract_comparison_sides(query)
        for side in comparison_sides:
            best = _find_best_doc_for_terms(docs, side, selected)
            if best:
                selected.append(best)
                source_counts[best.metadata.get("paper_id", "unknown")] += 1
            else:
                diagnostics.append(("missing_side", side))

    # 2. Add top relevant documents while allowing multiple chunks from strong sources
    for doc in docs:
        if len(selected) >= final_k:
            break
        if doc in selected:
            continue
        if _is_near_duplicate(doc, selected, threshold=0.92):
            diagnostics.append(("duplicate", doc))
            continue
        source_id = doc.metadata.get("paper_id", "unknown")
        if source_counts[source_id] < max_per_source:
            selected.append(doc)
            source_counts[source_id] += 1
        else:
            diagnostics.append(("source_cap", doc))

    # 3. If still under target, add more top docs even if source caps have been reached
    for doc in docs:
        if len(selected) >= final_k:
            break
        if doc in selected:
            continue
        if _is_near_duplicate(doc, selected, threshold=0.95):
            diagnostics.append(("duplicate_fallback", doc))
            continue
        selected.append(doc)

    return selected[:final_k], diagnostics


def _extract_comparison_sides(query: str) -> List[str]:
    """Extract the two comparison sides from a query for lightweight coverage balancing."""
    query_lower = query.lower().strip()
    query_lower = re.sub(r"[\-—]", " ", query_lower)

    # Common comparison patterns
    compare_patterns = [
        r"comparison of\s+(.+?)\s+and\s+(.+)",
        r"comparison between\s+(.+?)\s+and\s+(.+)",
        r"(.+?)\s+(?:vs|versus|v\.|compared to|compared with)\s+(.+)",
    ]
    for pattern in compare_patterns:
        match = re.search(pattern, query_lower)
        if match:
            left = match.group(1).strip()
            right = match.group(2).strip()
            if left and right:
                return [left, right]

    if " and " in query_lower and "comparison" in query_lower:
        parts = [part.strip() for part in query_lower.split(" and ") if part.strip()]
        if len(parts) >= 2:
            return parts[:2]

    if " or " in query_lower and "comparison" in query_lower:
        parts = [part.strip() for part in query_lower.split(" or ") if part.strip()]
        if len(parts) >= 2:
            return parts[:2]

    return []


def _find_best_doc_for_terms(docs: List, term: str, selected: List) -> Optional[object]:
    """Find the highest-ranked document that best matches a comparison side."""
    normalized_term = _normalize_text(term)
    term_tokens = [token for token in normalized_term.split() if len(token) > 2 and token not in {"comparison", "versus", "compared", "to", "and", "or", "of", "between"}]
    best_doc = None
    best_score = 0
    for doc in docs:
        if doc in selected:
            continue
        content = _normalize_text(doc.page_content)
        score = 0
        if normalized_term in content:
            score += 10
        for token in term_tokens:
            if token in content:
                score += 2
        if score > best_score:
            best_score = score
            best_doc = doc
    return best_doc if best_score > 0 else None


def _compute_term_coverage(docs: List, terms: List[str]) -> Dict[str, int]:
    """Compute how many retrieved docs cover each focus term or query side."""
    coverage = {}
    for term in terms:
        normalized_term = _normalize_text(term)
        term_tokens = [token for token in normalized_term.split() if token]
        count = 0
        for doc in docs:
            content = _normalize_text(doc.page_content)
            focus_hits = [
                _normalize_text(hit)
                for hit in doc.metadata.get("focus_hits", [])
                if isinstance(hit, str)
            ] if hasattr(doc, 'metadata') else []

            if normalized_term and normalized_term in content:
                count += 1
                continue
            if normalized_term and normalized_term in focus_hits:
                count += 1
                continue
            if term_tokens and all(token in content for token in term_tokens):
                count += 1
                continue
        coverage[term] = count
    return coverage


def _topic_aspect(doc) -> str:
    """Estimate the main topic aspect of a chunk for better topic diversity."""
    text = doc.page_content.lower()
    aspects = {
        "methods": {
            "algorithm", "method", "approach", "architecture", "policy",
            "optimization", "training", "model", "implementation", "technique",
            "parameter"
        },
        "applications": {
            "application", "use case", "robotics", "healthcare", "finance",
            "autonomous", "control", "navigation", "deployment", "dataset",
            "environment", "real-world"
        },
        "limitations": {
            "limitation", "challenge", "problem", "issue", "drawback",
            "constraint", "scalability", "overhead", "sample efficiency",
            "failure", "robustness"
        },
        "benchmarks": {
            "benchmark", "evaluation", "experiment", "performance", "accuracy",
            "metric", "result", "score", "baseline", "comparison", "sota",
            "state-of-the-art"
        },
        "future work": {
            "future work", "future", "direction", "outlook", "potential",
            "extension", "next step", "open problem", "roadmap", "prospect"
        }
    }

    scores = {name: 0 for name in aspects}
    for name, keywords in aspects.items():
        for keyword in keywords:
            if keyword in text:
                scores[name] += 1

    best_aspect = max(scores, key=scores.get)
    return best_aspect if scores[best_aspect] > 0 else "general"


def _deduplicate_candidates(docs: List, threshold: float = 0.78) -> List:
    """Remove near-duplicate candidate chunks before reranking."""
    unique = []
    for doc in docs:
        if not _is_near_duplicate(doc, unique, threshold=threshold):
            unique.append(doc)
    return unique


def _is_near_duplicate(doc, candidates: List, threshold: float = 0.78) -> bool:
    """Check whether a candidate chunk is nearly duplicate of any selected chunk."""
    content = _normalize_text(doc.page_content)
    for existing in candidates:
        other = _normalize_text(existing.page_content)
        if _text_similarity(content, other) >= threshold:
            return True
    return False


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def _text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 1.0 if a == b else 0.0
    set_a = set(a.split())
    set_b = set(b.split())
    jaccard = len(set_a & set_b) / max(1, len(set_a | set_b))
    seq = SequenceMatcher(None, a, b).ratio()
    return max(jaccard, seq)


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
    
    limitation_keywords = {
        "limitation", "challenge", "problem", "issue", "difficulty",
        "constraint", "weakness", "gap", "drawback", "fail", "failure",
        "error", "incomplete", "scalability", "performance", "trade-off",
        "trade-offs", "overhead"
    }
    
    technical_keywords = {
        "algorithm", "architecture", "framework", "method", "technique",
        "approach", "implementation", "system", "module", "component",
        "optimization", "efficiency", "complexity"
    }
    
    def score_doc(doc, terms, keywords):
        content = doc.page_content.lower()
        score = 0
        for term in terms:
            if term.lower() in content:
                score += 3
        for keyword in keywords:
            if keyword in content:
                score += 1
        return score
    
    if priority == "limitations":
        scored_docs = [(doc, score_doc(doc, focus_terms, limitation_keywords)) for doc in docs]
    elif priority == "technical":
        scored_docs = [(doc, score_doc(doc, focus_terms, technical_keywords)) for doc in docs]
    else:
        scored_docs = [(doc, score_doc(doc, focus_terms, set())) for doc in docs]
    
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs]


def rerank(query, docs):
    if not docs:
        return []
    passages = [doc.page_content for doc in docs]
    rerank_scores = reranker.predict([(query, passage) for passage in passages])
    reranked = sorted(zip(docs, rerank_scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked]
