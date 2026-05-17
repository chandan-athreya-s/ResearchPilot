import requests
import re

from app.services.query_analyzer import analyze_query

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
    params = {
        "search": build_search_query(query),
        "per-page": max_results,
        "filter": "is_oa:true,concepts.id:C41008148"
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json()

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

    return papers