import httpx
import logging
import requests
from typing import Any, Dict, List, Set

from app.agents.retrieval_agent import compute_relevance_score
from app.services.pdf_downloader import download_pdf_with_fallbacks
from app.services.pdf_extractor import extract_text_from_pdf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
OPENALEX_BASE_URL = "https://api.openalex.org/works"
MAX_RETRIES_DOWNLOAD = 2
MAX_RETRIES_API = 2
MAX_NEW_PAPERS = 5
REFS_PER_PAPER = 5  # Max cited papers to fetch per input paper


def augment_papers(
    papers: List[Dict],
    paper_ids_seen: set = None,
    cited_paper_ids: List[str] = None,
    primary_query: str = None,
    verbose: bool = False
) -> Dict:
    """
    Augment papers with citations from OpenAlex referenced_works (1-hop only).
    
    Workflow:
    1. For each input paper, query OpenAlex for referenced_works
    2. Extract cited paper IDs and fetch their full metadata
    3. Skip any papers already in input (strict deduplication by paper_id)
    4. Cap total new papers at MAX_NEW_PAPERS
    5. Attempt PDF download for new papers
    6. Return augmented papers list
    
    Args:
        papers: List of paper dicts from retrieval_agent
        verbose: Enable detailed logging
    
    Returns:
        Dict with keys: augmented_papers, new_papers_added
        Example: {
            "augmented_papers": [
                {...original papers...},
                {...new papers from citations...}
            ],
            "new_papers_added": 3
        }
    """
    if not papers:
        logger.info("Citation Agent: No papers to augment")
        return {
            "augmented_papers": papers,
            "new_papers_added": 0
        }
    
    if verbose:
        logger.info(f"Citation Agent: Processing {len(papers)} papers for citations")
    logger.debug(f"Citation agent received cited_paper_ids: {cited_paper_ids}")
    
    # Step 1: Build set of existing paper IDs to skip duplicates
    existing_paper_ids = paper_ids_seen if paper_ids_seen else {p.get("paper_id") for p in papers}
    normalized_papers = [_normalize_paper_metadata(p) for p in papers]

    # Step 2: Determine candidate cited works from structured citation IDs or paper references
    if cited_paper_ids:
        # FIX 4: Add debug logging at entry point
        logger.debug(f"Citation agent attempting to resolve: {cited_paper_ids}")
        requested_ids = []
        reference_ids_seen = set()
        for source_id in cited_paper_ids:
            # FIX 4: Normalize OpenAlex ID
            normalized_source_id = normalize_openalex_id(source_id)
            if not normalized_source_id or normalized_source_id in existing_paper_ids:
                continue
            try:
                source_refs = _fetch_referenced_works(normalized_source_id, verbose)
                for ref_id in source_refs:
                    if ref_id and ref_id not in existing_paper_ids and ref_id not in reference_ids_seen:
                        reference_ids_seen.add(ref_id)
                        requested_ids.append(ref_id)
                if verbose:
                    logger.info(f"Cited paper {normalized_source_id} contributed {len(source_refs)} references")
            except Exception as e:
                logger.warning(f"Failed to fetch references for cited paper {normalized_source_id}: {str(e)}")
        if verbose:
            logger.info(f"Citation Agent: resolved {len(requested_ids)} references from {len(cited_paper_ids)} cited paper IDs")
    else:
        requested_ids = _extract_cited_papers(normalized_papers, existing_paper_ids, verbose)

    if verbose:
        logger.info(f"Found {len(requested_ids)} candidate cited papers")
    else:
        logger.info(f"Citation Agent: Found {len(requested_ids)} candidate papers")

    # Step 3: Fetch full metadata for cited papers
    new_papers = _fetch_cited_papers_metadata(requested_ids, verbose)
    if primary_query:
        filtered_new_papers = []
        for paper in new_papers:
            score = compute_relevance_score(paper, primary_query)
            if score >= 0.08:
                filtered_new_papers.append(paper)
            elif verbose:
                logger.debug(f"Dropped new cited paper '{paper.get('title','Unknown')}' with relevance score {score:.3f}")
        new_papers = filtered_new_papers
        if verbose:
            logger.info(f"Filtered cited papers to {len(new_papers)} after primary query scoring")
    
    if verbose:
        logger.info(f"Fetched metadata for {len(new_papers)} cited papers")
    
    # Step 4: Cap at MAX_NEW_PAPERS
    capped_new_papers = new_papers[:MAX_NEW_PAPERS]
    
    if verbose and len(new_papers) > MAX_NEW_PAPERS:
        logger.info(f"Capped new papers: {len(new_papers)} → {len(capped_new_papers)}")
    
    # Step 5: Attempt PDF download for new papers
    enhanced_new_papers = _download_pdfs_for_new_papers(capped_new_papers, verbose)
    
    # Step 6: Combine original + new papers
    augmented_papers = papers + enhanced_new_papers
    
    if verbose:
        logger.info(f"Citation augmentation complete: {len(papers)} + {len(enhanced_new_papers)} = {len(augmented_papers)} total papers")
    else:
        logger.info(f"Citation Agent: Augmented with {len(enhanced_new_papers)} additional papers")
    
    return {
        "augmented_papers": augmented_papers,
        "new_papers_added": len(enhanced_new_papers)
    }


def _normalize_paper_metadata(paper: Dict) -> Dict:
    """Normalize paper metadata so citation augmentation creates consistent objects."""
    return {
        "paper_id": paper.get("paper_id") or paper.get("id") or "",
        "title": paper.get("title", "Unknown Title"),
        "abstract": paper.get("abstract", ""),
        "url": paper.get("url") or paper.get("id") or "",
        "arxiv_id": paper.get("arxiv_id"),
        "ids": paper.get("ids", {}),
        "locations": paper.get("locations", []),
        "authors": paper.get("authors", []),
        "year": paper.get("year"),
        "venue": paper.get("venue"),
        "doi": paper.get("doi"),
        "pdf_url": paper.get("pdf_url"),
        "open_access": paper.get("open_access", {}),
        "document_type": paper.get("document_type", "pdf"),
        "keywords": paper.get("keywords", []),
        "full_text": paper.get("full_text", "")
    }


def _extract_cited_papers(
    papers: List[Dict],
    existing_paper_ids: Set[str],
    verbose: bool
) -> List[str]:
    """
    Extract unique cited paper IDs from all input papers.
    
    Args:
        papers: List of paper dicts
        existing_paper_ids: Set of paper IDs already in input (to skip duplicates)
        verbose: Enable detailed logging
    
    Returns:
        List of unique cited paper IDs (not in existing_paper_ids)
    """
    cited_ids_set = set()
    failed_papers = 0
    skipped_papers = 0
    
    for paper in papers:
        paper_id = paper.get("paper_id")
        paper_title = paper.get("title", "Unknown")
        
        if not paper_id:
            skipped_papers += 1
            logger.warning(f"Citation Agent: Skipping malformed paper record without paper_id: '{paper_title}'")
            continue

        try:
            ref_ids = _fetch_referenced_works(paper_id, verbose)
            new_refs = [ref_id for ref_id in ref_ids if ref_id and ref_id not in existing_paper_ids]
            if verbose:
                logger.info(f"Found {len(new_refs)} new citations for '{paper_title}'")
            cited_ids_set.update(new_refs)
        except Exception as e:
            failed_papers += 1
            logger.warning(f"Failed to fetch references for '{paper_title}': {str(e)}")

    if failed_papers > 0:
        logger.info(f"Failed to fetch references from {failed_papers}/{len(papers)} papers")
    if skipped_papers > 0:
        logger.info(f"Skipped {skipped_papers} malformed source papers")
    
    # Return as list, maintaining reasonable order
    return list(cited_ids_set)


def normalize_openalex_id(raw_id: str) -> str:
    """Normalize OpenAlex ID to full URL format."""
    if raw_id.startswith("https://openalex.org/"):
        return raw_id
    if raw_id.startswith("W") and raw_id[1:].isdigit():
        return f"https://openalex.org/{raw_id}"
    return raw_id


def _fetch_referenced_works(paper_id: str, verbose: bool) -> List[str]:
    """
    Query OpenAlex API for referenced_works and fall back to related_works.
    
    Args:
        paper_id: OpenAlex paper ID
        verbose: Enable detailed logging
    
    Returns:
        List of referenced paper IDs (capped at REFS_PER_PAPER)
    """
    url = f"{OPENALEX_BASE_URL}/{paper_id}"
    try:
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            resp = client.get(url, params={"select": "referenced_works,id,title,publication_year"})
            resp.raise_for_status()
            data = resp.json()
            refs = data.get("referenced_works", []) or []

            # FIX 4: Add debug logging after fetch
            logger.debug(f"OpenAlex referenced_works for {paper_id}: {len(refs)} entries")
            if len(refs) == 0:
                logger.warning(f"Empty referenced_works for {paper_id}. Trying related_works fallback.")
                # Fallback:
                fallback_resp = client.get(url, params={"select": "related_works,id"})
                fallback_resp.raise_for_status()
                refs = fallback_resp.json().get("related_works", []) or []
                logger.info(f"related_works fallback returned {len(refs)} entries for {paper_id}")

            # FIX 5: Year floor filter and hard cap
            MIN_PUBLICATION_YEAR = 2010
            if 'publication_year' in data:  # Only filter if we have year data (referenced_works)
                candidates = [
                    ref for ref in refs
                    if isinstance(ref, dict) and ref.get('publication_year', 9999) >= MIN_PUBLICATION_YEAR
                ]
                logger.info(f"Year filter: {len(refs)} → {len(candidates)} refs (floor={MIN_PUBLICATION_YEAR})")
                ref_ids = [ref.get('id') for ref in candidates if isinstance(ref, dict) and ref.get('id')]
            else:
                # For related_works fallback, refs are already IDs
                if isinstance(refs, list) and refs and isinstance(refs[0], str):
                    ref_ids = refs  # Already IDs
                else:
                    ref_ids = [ref.get('id') for ref in refs if isinstance(ref, dict) and ref.get('id')]
            
            # Hard cap at 10 candidates per source paper
            ref_ids = ref_ids[:10]
            
            if verbose:
                logger.info(f"Citation Agent: parsed {len(ref_ids)} references for {paper_id}")
            return ref_ids[:REFS_PER_PAPER]
    except httpx.TimeoutException:
        logger.warning(f"Timeout fetching referenced_works for {paper_id}")
        return []
    except Exception as e:
        logger.warning(f"Failed to fetch referenced_works for {paper_id}: {e}")
        return []


def _fetch_cited_by_ids(cited_by_url: str, verbose: bool) -> List[str]:
    """Fetch paper IDs from an OpenAlex cited_by_api_url endpoint."""
    try:
        data = _openalex_get_json(cited_by_url, {"per-page": REFS_PER_PAPER}, verbose)
        results = data.get("results", []) if isinstance(data, dict) else []
        ids = [item.get("id") for item in results if isinstance(item, dict) and item.get("id")]
        if verbose:
            logger.info(f"Citation Agent: found {len(ids)} cited_by references")
        return ids[:REFS_PER_PAPER]
    except Exception as e:
        logger.warning(f"Citation Agent: failed to fetch cited_by_api_url: {str(e)}")
        return []


def _openalex_get_json(url: str, params: dict, verbose: bool) -> Dict:
    """GET JSON from OpenAlex with retry handling."""
    last_error = None
    for attempt in range(MAX_RETRIES_API + 1):
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            last_error = e
            if verbose:
                logger.warning(f"OpenAlex request failed for {url} (attempt {attempt + 1}): {str(e)}")
            if attempt < MAX_RETRIES_API:
                continue
    raise Exception(f"OpenAlex API error for {url}: {str(last_error)}")


def _fetch_cited_papers_metadata(cited_paper_ids: List[str], verbose: bool) -> List[Dict]:
    """
    Fetch full metadata for list of cited paper IDs from OpenAlex.
    
    Args:
        cited_paper_ids: List of OpenAlex paper IDs
        verbose: Enable detailed logging
    
    Returns:
        List of paper dicts with metadata
    """
    papers = []
    failed_fetches = 0
    
    for paper_id in cited_paper_ids:
        try:
            # Query OpenAlex for full paper metadata
            paper_dict = _fetch_paper_metadata(paper_id)
            papers.append(paper_dict)
            
            if verbose:
                logger.info(f"Fetched metadata for '{paper_dict.get('title', 'Unknown')}'")
        
        except Exception as e:
            failed_fetches += 1
            if verbose:
                logger.warning(f"Failed to fetch metadata for {paper_id}: {str(e)}")
    
    if failed_fetches > 0:
        logger.info(f"Failed to fetch metadata for {failed_fetches}/{len(cited_paper_ids)} papers")
    
    return papers


def _fetch_paper_metadata(paper_id: str) -> Dict:
    """
    Query OpenAlex API for full paper metadata.
    
    Args:
        paper_id: OpenAlex paper ID
    
    Returns:
        Paper dict with metadata (matches retrieval_agent paper format)
    """
    from app.services.openalex_client import reconstruct_abstract, extract_arxiv_id
    
    url = f"{OPENALEX_BASE_URL}/{paper_id}"
    item = _openalex_get_json(url, None, verbose=False)
    
    abstract = reconstruct_abstract(item.get("abstract_inverted_index"))
    arxiv_id = extract_arxiv_id(item.get("locations", []))
    keywords = [concept.get("display_name") for concept in item.get("concepts", []) if isinstance(concept, dict) and concept.get("display_name")]
    
    authors = [
        auth["author"]["display_name"]
        for auth in item.get("authorships", [])
        if isinstance(auth, dict) and auth.get("author", {}).get("display_name")
    ][:3]
    year = item.get("publication_year")
    venue = item.get("host_venue", {}).get("display_name")
    doi = item.get("doi")
    pdf_url = None
    for location in item.get("locations", []):
        if location.get("pdf_url"):
            pdf_url = location["pdf_url"]
            break
    
    return {
        "paper_id": item.get("id"),
        "title": item.get("title", "Unknown Title"),
        "abstract": abstract,
        "url": item.get("id"),
        "arxiv_id": arxiv_id,
        "ids": item.get("ids", {}),
        "locations": item.get("locations", []),
        "authors": authors,
        "year": year,
        "venue": venue,
        "doi": doi,
        "pdf_url": pdf_url,
        "open_access": item.get("open_access", {}),
        "keywords": keywords[:5],
        "full_text": "",
        "document_type": "abstract_only"
    }


def _download_pdfs_for_new_papers(papers: List[Dict], verbose: bool) -> List[Dict]:
    """
    Attempt PDF download + text extraction for new papers (with retry logic).
    
    Graceful degradation: keep paper even if PDF fetch fails.
    
    Args:
        papers: List of new paper dicts
        verbose: Enable detailed logging
    
    Returns:
        Papers with "full_text" field added (empty string if extraction failed)
    """
    papers_with_text = []
    success_count = 0
    
    for paper in papers:
        paper_title = paper.get("title", "Unknown")
        
        # Try to download PDF with retries
        pdf_path = None
        for attempt in range(MAX_RETRIES_DOWNLOAD + 1):
            try:
                pdf_path = download_pdf_with_fallbacks(paper)
                if pdf_path:
                    if verbose:
                        logger.info(f"Downloaded PDF for cited paper '{paper_title}'")
                    break
            except Exception as e:
                if verbose:
                    logger.warning(f"PDF download failed for '{paper_title}' (attempt {attempt + 1}): {str(e)}")
                
                if attempt < MAX_RETRIES_DOWNLOAD:
                    continue
                else:
                    logger.warning(f"PDF download failed for '{paper_title}' after {MAX_RETRIES_DOWNLOAD + 1} retries")
        
        # Extract text if PDF was downloaded
        full_text = ""
        document_type = paper.get("document_type", "pdf")
        if pdf_path:
            try:
                full_text = extract_text_from_pdf(pdf_path)
                if full_text:
                    success_count += 1
                    if verbose:
                        logger.info(f"Extracted {len(full_text)} chars from cited paper '{paper_title}'")
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
                logger.info(f"No PDF available for cited paper '{paper_title}', will use abstract-only fallback")
            document_type = "abstract_only"

        if not full_text:
            full_text = _build_abstract_fallback_text(paper)

        if full_text and document_type == "abstract_only":
            paper["document_type"] = "abstract_only"
        else:
            paper["document_type"] = document_type

        paper["full_text"] = full_text
        papers_with_text.append(paper)
    
    logger.info(f"Successfully extracted text from {success_count}/{len(papers)} cited papers")
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
