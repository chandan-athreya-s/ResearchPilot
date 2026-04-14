import requests

BASE_URL = "https://api.openalex.org/works"


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


def fetch_papers(query, max_results=20):
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

        papers.append({
            "title": item.get("title"),
            "abstract": abstract,
            "url": item.get("id")
        })

    return papers