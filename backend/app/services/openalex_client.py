import requests
import re

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


def fetch_papers(query, max_results=8):
    params = {
        "search": query,
        "per-page": max_results
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
            "authors": authors,
            "year": year,
            "venue": venue,
            "doi": doi,
            "pdf_url": pdf_url
        })

    return papers