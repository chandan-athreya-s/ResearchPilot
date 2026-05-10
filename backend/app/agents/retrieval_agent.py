import logging
import re
from typing import Dict, List, Set
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.services.openalex_client import fetch_papers
from app.services.pdf_downloader import download_pdf_with_fallbacks
from app.services.pdf_extractor import extract_text_from_pdf
from app.services.text_processor import process_documents
from app.services.vector_store import create_vector_store

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_PAPERS_TOTAL = 15
MIN_TEXT_LENGTH = 500
MAX_RETRIES_DOWNLOAD = 2

# FIX 3: Relevance filter improvements
RELEVANCE_THRESHOLD = 0.10  # lowered from 0.12 for better recall
DOMAIN_BLOCKLIST_PHRASES = [
    "food fraud", "spectroscopy", "industry 4.0", "signal peptide",
    "physionet", "men who have sex with men", "dental domain",
    "protein language", "model inversion attack"
]


def is_blocklisted(paper: dict) -> bool:
    """Check if paper should be blocked based on domain-specific phrases."""
    text = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()
    return any(phrase in text for phrase in DOMAIN_BLOCKLIST_PHRASES)


def compute_relevance_score(paper: dict, query: str) -> float:
    """Compute relevance score for a paper against a single query."""
    query_tokens = _tokenize_text(query)
    if not query_tokens:
        return 0.0
    
    title = paper.get("title", "").lower()
    abstract = paper.get("abstract", "").lower()
    keyword_text = " ".join(paper.get("keywords", [])).lower()
    content = " ".join([title, abstract, keyword_text])
    
    overlap = len(query_tokens.intersection(_tokenize_text(content)))
    score = overlap / len(query_tokens)
    
    # Bonus for keyword matches
    normalized_keywords = {kw.lower() for kw in paper.get("keywords", [])}
    if normalized_keywords and any(kw in query.lower() for kw in normalized_keywords):
        score += 0.1
    
    return min(score, 1.0)


def compute_best_relevance_score(paper: dict, primary_query: str, sub_queries: list[str]) -> tuple[float, str]:
    """Compute best relevance score across primary and sub-queries."""
    scores = [(compute_relevance_score(paper, primary_query), primary_query)]
    for sq in sub_queries:
        scores.append((compute_relevance_score(paper, sq), sq))
    best_score, winning_query = max(scores, key=lambda x: x[0])
    return best_score, winning_query


def retrieve_papers(
    sub_queries: List[str],
    keywords: List[str],
    primary_query: str,
    paper_ids_seen: set = None,
    verbose: bool = False
) -> Dict:
    """
    Retrieve papers from OpenAlex, process PDFs, and build FAISS index.
    
    Args:
        sub_queries: List of 2-4 sub-queries from query_agent
        keywords: List of 3-5 keywords from query_agent
        verbose: Enable detailed logging
    
    Returns:
        Dict with keys: papers, chunks, vector_store, metadata
        Example: {
            "papers": [
                {
                    "paper_id": "...",
                    "title": "...",
                    "full_text": "extracted text from PDF",
                    ...other metadata...
                }
            ],
            "chunks": List[Document],
            "vector_store": FAISS,
            "metadata": {
                "papers_fetched": 10,
                "papers_with_text": 8,
                "chunks_created": 45
            }
        }
    """
    if verbose:
        logger.info(f"AGENT: Processing {len(sub_queries)} sub-queries, {len(keywords)} keywords")
    
    # Step 1: Fetch papers from OpenAlex
    all_papers = _fetch_papers_with_retry(sub_queries, verbose)
    
    # Step 2: Filter out already seen papers
    if paper_ids_seen:
        all_papers = [p for p in all_papers if p['paper_id'] not in paper_ids_seen]
        if verbose:
            logger.info(f"Filtered out {len(paper_ids_seen)} already seen papers")
    
    # Step 3: Merge and deduplicate
    unique_papers = _deduplicate_papers(all_papers, verbose)

    # FIX 3: Apply domain blocklist before relevance check
    original_count = len(unique_papers)
    unique_papers = [p for p in unique_papers if not is_blocklisted(p)]
    logger.info(f"Blocklist filtered {original_count - len(unique_papers)} papers")

    # Step 4: Filter by semantic relevance to primary query
    relevant_papers = []
    for paper in unique_papers:
        score, winning_query = compute_best_relevance_score(paper, primary_query, sub_queries)
        if score >= RELEVANCE_THRESHOLD:
            relevant_papers.append(paper)
            logger.debug(f"Kept '{paper.get('title','Unknown')[:60]}' — best score {score:.3f} on '{winning_query}'")
        else:
            logger.debug(f"Filtered out paper '{paper.get('title','Unknown')}' with best score {score:.3f} on '{winning_query}'")

    if verbose:
        logger.info(f"Retained {len(relevant_papers)}/{len(unique_papers)} papers after primary-query relevance filter")
    else:
        logger.info(f"Retrieval Agent: relevance filtered to {len(relevant_papers)} papers")
    
    # Step 5: Cap at MAX_PAPERS_TOTAL
    capped_papers = relevant_papers[:MAX_PAPERS_TOTAL]
    if verbose:
        logger.info(f"Capped papers to {len(capped_papers)} (max {MAX_PAPERS_TOTAL})")
    else:
        logger.info(f"Retrieval Agent: Deduplicated to {len(capped_papers)} papers")
    
    # Step 4: Download PDFs and extract text, with abstract-only fallbacks
    papers_with_text = _download_and_extract_pdfs(capped_papers, verbose)
    
    # Step 5: Filter by minimum text length and relevance
    filtered_papers = [
        p for p in papers_with_text 
        if len(p.get("full_text", "")) >= MIN_TEXT_LENGTH or p.get("document_type") == "abstract_only"
    ]

    if verbose:
        logger.info(f"Filtered to {len(filtered_papers)} candidate papers after extraction/fallback")
    else:
        logger.info(f"Retrieved {len(filtered_papers)} candidate papers")

    filtered_papers = _filter_papers_by_relevance(filtered_papers, sub_queries, keywords, verbose)

    if verbose:
        logger.info(f"Retained {len(filtered_papers)} papers after relevance filtering")

    # Step 6: Process into chunks
    chunks = process_documents(filtered_papers)
    
    if verbose:
        logger.info(f"Created {len(chunks)} chunks from {len(filtered_papers)} papers")
    
    # Step 7: Build FAISS index
    if chunks:
        vector_store = create_vector_store(chunks)
        if verbose:
            logger.info("FAISS index created and saved")
    else:
        logger.warning("No chunks to index - vector store creation skipped")
        vector_store = None
    
    # Prepare metadata store for later use in llm_service
    metadata_store = {p["paper_id"]: p for p in filtered_papers}
    papers_with_extracted_text_set = {p["paper_id"] for p in filtered_papers}
    
    return {
        "papers": filtered_papers,
        "chunks": chunks,
        "vector_store": vector_store,
        "metadata": {
            "papers_fetched": len(all_papers),
            "papers_after_dedup": len(unique_papers),
            "papers_capped": len(capped_papers),
            "papers_with_text": len(filtered_papers),
            "chunks_created": len(chunks)
        },
        "metadata_store": metadata_store,
        "papers_with_extracted_text": papers_with_extracted_text_set
    }


def _fetch_papers_with_retry(sub_queries: List[str], verbose: bool) -> List[Dict]:
    """
    Fetch papers from OpenAlex for each sub-query with retry logic.
    
    Args:
        sub_queries: List of queries to search
        verbose: Enable detailed logging
    
    Returns:
        List of paper dicts (may contain duplicates)
    """
    all_papers = []
    
    for query in sub_queries:
        for attempt in range(MAX_RETRIES_DOWNLOAD + 1):
            try:
                if verbose and attempt > 0:
                    logger.info(f"Retrying OpenAlex fetch for '{query}' (attempt {attempt})")
                
                papers = fetch_papers(query, max_results=5)
                for paper in papers:
                    if not paper.get("source_sub_queries"):
                        paper["source_sub_queries"] = [query]
                    elif query not in paper["source_sub_queries"]:
                        paper["source_sub_queries"].append(query)
                all_papers.extend(papers)
                if verbose:
                    logger.info(f"Fetched {len(papers)} papers for '{query}'")
                break
            except Exception as e:
                if verbose:
                    logger.warning(f"OpenAlex fetch failed for '{query}' (attempt {attempt + 1}): {str(e)}")
                
                if attempt < MAX_RETRIES_DOWNLOAD:
                    continue
                else:
                    logger.warning(f"All retries exhausted for query '{query}'")
    
    logger.info(f"Total papers fetched: {len(all_papers)}")
    return all_papers


def normalize_title(title: str) -> str:
    """Normalize paper titles for deduplication and fuzzy matching."""
    import re
    if not title:
        return ""
    cleaned = re.sub(r"[\W_]+", " ", title).strip().lower()
    collapsed = re.sub(r"\s+", " ", cleaned)
    return collapsed


def add_chunks_to_index(new_chunks: list[dict]) -> None:
    """Add new chunks to existing FAISS index without full rebuild."""
    if not new_chunks:
        return
    # Note: This is a placeholder implementation
    # Actual FAISS incremental add would require access to existing index
    logger.info(f"Placeholder: Would add {len(new_chunks)} chunks to FAISS index")
    """Compute a relevance score between the paper and the primary query using TF-IDF cosine similarity."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        logger.warning("sklearn not available, defaulting relevance score to 0.5")
        return 0.5

    text_fields = [paper.get("title", ""), paper.get("abstract", ""), " ".join(paper.get("keywords", []) or [])]
    document_text = " ".join([field for field in text_fields if field]).strip()
    if not document_text or not primary_query:
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([primary_query, document_text])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return float(score)


def _deduplicate_papers(papers: List[Dict], verbose: bool) -> List[Dict]:
    """
    Deduplicate papers by normalized title and then by paper_id.
    
    Args:
        papers: List of paper dicts
        verbose: Enable detailed logging
    
    Returns:
        Deduplicated list of paper dicts
    """
    seen_titles = set()
    candidate_papers = []
    for paper in papers:
        title_norm = normalize_title(paper.get("title", ""))
        if title_norm and title_norm not in seen_titles:
            seen_titles.add(title_norm)
            candidate_papers.append(paper)
        elif not title_norm:
            candidate_papers.append(paper)

    seen_ids = set()
    unique_papers = []
    for paper in candidate_papers:
        paper_id = paper.get("paper_id")
        if paper_id not in seen_ids:
            seen_ids.add(paper_id)
            unique_papers.append(paper)

    if verbose:
        logger.info(f"Deduplicated by title+id: {len(papers)} → {len(unique_papers)} unique papers")
    
    return unique_papers


def _download_and_extract_pdfs(papers: List[Dict], verbose: bool) -> List[Dict]:
    """
    Download PDFs and extract text for each paper with retry logic.
    
    Args:
        papers: List of paper dicts
        verbose: Enable detailed logging
    
    Returns:
        List of paper dicts with "full_text" field added (empty string if extraction failed)
    """
    papers_with_text = []
    success_count = 0
    fallback_count = 0
    
    for i, paper in enumerate(papers):
        paper_title = paper.get("title", "Unknown")

        pdf_path = None
        try:
            pdf_path = download_pdf_with_fallbacks(paper)
            if pdf_path and verbose:
                logger.info(f"Downloaded PDF for '{paper_title}'")
        except Exception as e:
            logger.warning(f"PDF download failed for '{paper_title}': {str(e)}")

        # Extract text if PDF was downloaded
        full_text = ""
        document_type = paper.get("document_type", "pdf")
        if pdf_path:
            try:
                full_text = extract_text_from_pdf(pdf_path)
                if full_text:
                    success_count += 1
                    if verbose:
                        logger.info(f"Extracted {len(full_text)} chars from '{paper_title}'")
                    document_type = "pdf"
                else:
                    if verbose:
                        logger.info(f"PDF extracted but text is empty for '{paper_title}', falling back to abstract-only content")
            except Exception as e:
                logger.warning(f"Text extraction failed for '{paper_title}': {str(e)}")
                full_text = ""
                document_type = "abstract_only"
        else:
            if verbose:
                logger.info(f"No PDF available for '{paper_title}', will use abstract-only fallback")
            document_type = "abstract_only"

        if not full_text:
            fallback_text = _build_abstract_fallback_text(paper)
            if fallback_text:
                full_text = fallback_text
                fallback_count += 1
                document_type = "abstract_only"
                if verbose:
                    logger.info(f"Using abstract fallback for '{paper_title}'")
            else:
                if verbose:
                    logger.debug(f"Discarding paper '{paper_title}' because neither PDF nor abstract content was available")
                continue

        if document_type == "abstract_only":
            paper["document_type"] = "abstract_only"
        else:
            paper["document_type"] = document_type

        # Add full_text to paper dict (even if fallback text was used)
        paper["full_text"] = full_text
        papers_with_text.append(paper)
    
    logger.info(f"Successfully extracted text from {success_count}/{len(papers)} papers")
    logger.info(f"Abstract fallback used for {fallback_count}/{len(papers)} papers")
    return papers_with_text


def _build_abstract_fallback_text(paper: Dict) -> str:
    """Generate pseudo-document text from title, abstract, and keywords."""
    abstract = paper.get("abstract", "")
    if not abstract:
        return ""

    title = paper.get("title", "Unknown Title")
    keywords = paper.get("keywords", []) or []
    keyword_str = ", ".join(keywords)
    sections = [title, abstract]
    if keyword_str:
        sections.append(f"Keywords: {keyword_str}")
    return "\n\n".join(sections)


def _filter_papers_by_relevance(
    papers: List[Dict],
    sub_queries: List[str],
    keywords: List[str],
    verbose: bool
) -> List[Dict]:
    """Filter weakly relevant papers before chunking while preserving diversity."""
    if not papers:
        return []

    normalized_keywords = {kw.lower() for kw in keywords if kw}
    source_query_text = " ".join(sub_queries).lower()
    query_tokens = _tokenize_text(source_query_text)
    scored = []

    for paper in papers:
        title = paper.get("title", "").lower()
        abstract = paper.get("abstract", "").lower()
        keyword_text = " ".join(paper.get("keywords", [])).lower()
        content = " ".join([title, abstract, keyword_text])

        overlap = len(query_tokens.intersection(_tokenize_text(content)))
        score = overlap / max(1, len(query_tokens))

        if normalized_keywords and any(kw in content for kw in normalized_keywords):
            score += 0.1
        if any(sub_query.lower() in content for sub_query in sub_queries):
            score += 0.05

        score = min(score, 1.0)
        scored.append((score, paper))

    scored.sort(key=lambda x: x[0], reverse=True)

    threshold = 0.25
    keep = [paper for score, paper in scored if score >= threshold]

    # Preserve coverage across sub-queries by keeping at least one paper for each sub-query
    for sub_query in sub_queries:
        if not any(sub_query.lower() in " ".join(paper.get("source_sub_queries", [])).lower() for paper in keep):
            candidates = [paper for score, paper in scored if sub_query.lower() in " ".join(paper.get("source_sub_queries", [])).lower()]
            if candidates:
                keep.append(candidates[0])
                if verbose:
                    logger.info(f"Preserving diversity: kept one paper for sub-query '{sub_query}'")

    # Ensure minimum retention for coverage
    if len(keep) < 6:
        for score, paper in scored:
            if paper not in keep:
                keep.append(paper)
            if len(keep) >= 6:
                break

    kept_ids = {paper["paper_id"] for paper in keep}
    filtered_papers = []
    for score, paper in scored:
        if paper["paper_id"] in kept_ids:
            filtered_papers.append(paper)
        elif verbose:
            reason = "low relevance"
            logger.info(f"Filtered out paper '{paper.get('title', 'Unknown')}' with score {score:.2f} ({reason})")

    return filtered_papers


def _tokenize_text(text: str) -> Set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token}

