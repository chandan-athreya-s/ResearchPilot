import requests
import re
import logging
import json

# Configure logging
logger = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org/works"


def extract_arxiv_id(locations):
    try:
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
    except Exception as e:
        logger.warning(f"Error extracting arXiv ID: {str(e)}")
    return None


def reconstruct_abstract(inverted_index):
    try:
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
    except Exception as e:
        logger.warning(f"Error reconstructing abstract: {str(e)}")
        return ""


def fetch_papers(query, max_results=8):
    try:
        params = {
            "search": query,
            "per-page": max_results,
            "filter": "is_oa:true,concepts.id:C41008148"
        }

        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()  # Raise exception for bad status codes
        
        try:
            data = response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            return []
            
    except requests.RequestException as e:
        logger.error(f"OpenAlex API request failed: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in fetch_papers: {str(e)}")
        return []

    papers = []

    try:
        for item in data.get("results", []):
            try:
                #FIX: use inverted index instead of abstract
                abstract = reconstruct_abstract(
                    item.get("abstract_inverted_index")
                )

                arxiv_id = extract_arxiv_id(item.get("locations", []))

                # Extract additional metadata with fallbacks
                authors = []
                try:
                    authors = [
                        auth["author"]["display_name"]
                        for auth in item.get("authorships", [])
                        if auth.get("author", {}).get("display_name")
                    ]
                    authors = authors[:3]
                except (KeyError, TypeError):
                    authors = []

                year = item.get("publication_year")
                venue = None
                try:
                    venue = item.get("host_venue", {}).get("display_name")
                except (KeyError, TypeError):
                    venue = None

                doi = item.get("doi")
                pdf_url = None
                try:
                    for location in item.get("locations", []):
                        if location.get("pdf_url"):
                            pdf_url = location["pdf_url"]
                            break
                except (KeyError, TypeError):
                    pdf_url = None

                keywords = []
                try:
                    for concept in item.get("concepts", []):
                        name = concept.get("display_name")
                        if name:
                            keywords.append(name)
                except (KeyError, TypeError):
                    keywords = []

                papers.append({
                    "paper_id": item.get("id"),  # OpenAlex ID as unique identifier
                    "title": item.get("title", "Unknown Title"),
                    "abstract": abstract,
                    "url": item.get("id", ""),
                    "arxiv_id": arxiv_id,
                    "ids": item.get("ids", {}),
                    "locations": item.get("locations", []),
                    "authors": authors,
                    "year": year,
                    "venue": venue,
                    "doi": doi,
                    "pdf_url": pdf_url,
                    "open_access": item.get("open_access", {}),
                    "keywords": keywords[:5]
                })
            except Exception as e:
                logger.warning(f"Error processing paper item: {str(e)}, skipping")
                continue
                
    except Exception as e:
        logger.error(f"Error processing results: {str(e)}")
        return papers  # Return any successfully processed papers

    return papers