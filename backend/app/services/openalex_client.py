import requests
import re

from app.services.query_analyzer import analyze_query
from app.services.rate_limit_manager import get_rate_limit_manager
from app.services.retrieval_cache import get_retrieval_cache


def _debug_response(response: requests.Response, context: str) -> None:
    print(f"[{context}] STATUS: {response.status_code}")
    print(f"[{context}] HEADERS: {dict(response.headers)}")
    body = getattr(response, "text", "") or ""
    print(f"[{context}] BODY: {body[:500]}")

BASE_URL = "https://api.openalex.org/works"


def extract_arxiv_id(locations):
    for location in locations:
        pdf_url = location.get("pdf_url")
        if pdf_url and "arxiv.org" in pdf_url:
            # Extract ID from https://arxiv.org/pdf/XXXX.XXXXX.pdf
            match = re.search(r'arxiv\.org/pdf/([^/]+)\.pdf', pdf_url)
            if match:
                return match.group(1)
        landing_page_url = location.get("landing_page_url")
        if landing_page_url and "arxiv.org" in landing_page_url:
            # Extract ID from https://arxiv.org/abs/XXXX.XXXXX
            match = re.search(r'arxiv\.org/abs/([^/]+)', landing_page_url)
            if match:
                return match.group(1)
    return None


def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return ""

    word_positions = {}

    # Build position → word mapping
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions[pos] = word

    # Sort words by position
    ordered_words = [word_positions[i] for i in sorted(word_positions)]

    return " ".join(ordered_words)


def build_search_query(query: str) -> str:
    analysis = analyze_query(query)
    focus_terms = analysis.get("focus_terms", [])
    comparison_terms = [term for pair in analysis.get("comparison_pairs", []) for term in pair]
    normalized_terms = []
    for term in focus_terms + comparison_terms:
        term = term.strip()
        if term and term not in normalized_terms:
            normalized_terms.append(term)

    additional_tokens = " ".join(normalized_terms)
    expanded_query = f"{query} {additional_tokens}".strip()
    return re.sub(r"\s+", " ", expanded_query)


def fetch_papers(query, max_results=8):
    rate_limit_mgr = get_rate_limit_manager()
    cache = get_retrieval_cache()
    cached = cache.get(query, "openalex")
    if cached is not None:
        print(f"[OpenAlex] Cache hit for query: {query[:50]}...")
        return cached

    if not rate_limit_mgr.is_enabled("openalex"):
        remaining = rate_limit_mgr.get_time_until_retry("openalex")
        print(f"[OpenAlex] Rate limited. Retry in {remaining:.1f}s")
        return []

    params = {
        "search": build_search_query(query),
        "per-page": max_results,
        "filter": "is_oa:true,concepts.id:C41008148"
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=12)
    except requests.exceptions.RequestException as exc:
        print(f"[OpenAlex] network failure: {exc}")
        rate_limit_mgr.record_network_error("openalex")
        return []

    if response.status_code == 429:
        retry_after = None
        if "retry-after" in response.headers:
            try:
                retry_after = int(response.headers["retry-after"])
            except ValueError:
                pass
        rate_limit_mgr.record_rate_limit("openalex", retry_after)
        return []

    if response.status_code != 200:
        _debug_response(response, "OpenAlex non-200 response")
        rate_limit_mgr.record_network_error("openalex")
        return []

    text = getattr(response, "text", "") or ""
    if not text.strip():
        print("[OpenAlex] empty response body")
        return []

    if "<html" in text.lower() or "<!doctype html" in text.lower():
        _debug_response(response, "OpenAlex HTML error response")
        return []

    try:
        data = response.json()
    except Exception as exc:
        print("OpenAlex JSON PARSE FAILED")
        _debug_response(response, "OpenAlex malformed JSON")
        print(f"[OpenAlex] parse exception: {exc}")
        rate_limit_mgr.record_network_error("openalex")
        return []

    if not isinstance(data, dict):
        print("[OpenAlex] unexpected response format, expected JSON object")
        _debug_response(response, "OpenAlex unexpected JSON format")
        rate_limit_mgr.record_network_error("openalex")
        return []

    papers = []

    for item in data.get("results", []):
        #FIX: use inverted index instead of abstract
        abstract = reconstruct_abstract(
            item.get("abstract_inverted_index")
        )

        arxiv_id = extract_arxiv_id(item.get("locations", []))

        # Extract additional metadata
        authors = [auth["author"]["display_name"] for auth in item.get("authorships", [])]
        year = item.get("publication_year")
        venue = item.get("host_venue", {}).get("display_name")
        doi = item.get("doi")
        pdf_url = None
        for location in item.get("locations", []):
            if location.get("pdf_url"):
                pdf_url = location["pdf_url"]
                break

        papers.append({
            "paper_id": item.get("id"),  # OpenAlex ID as unique identifier
            "title": item.get("title"),
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
            "open_access": item.get("open_access", {})
        })

    if papers:
        cache.set(query, "openalex", papers)
        rate_limit_mgr.record_success("openalex")
        print(f"[OpenAlex] Successfully fetched {len(papers)} papers")

    return papers