import requests
from pathlib import Path
from bs4 import BeautifulSoup
import re

PDF_DIR = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    # Remove scripts and styles
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text

def download_pdf_with_fallbacks(paper):
    paper_id = paper["paper_id"]
    title = paper["title"]
    arxiv_id = paper.get("arxiv_id")
    doi = paper.get("doi")
    ids = paper.get("ids", {})
    
    failed_sources = set()
    
    # Use OpenAlex IDs arxiv field if arXiv ID wasn't inferred from locations.
    if not arxiv_id:
        arxiv_id = ids.get("arxiv")

    if not arxiv_id:
        locations = paper.get("locations", [])
        for location in locations:
            landing_page_url = location.get("landing_page_url") or ""
            if "arxiv.org" in landing_page_url:
                match = re.search(r'arxiv\.org/abs/([^/]+)', landing_page_url)
                if match:
                    arxiv_id = match.group(1)
                    break
    
    # 1. arXiv direct PDF
    if arxiv_id:
        source_name = "arXiv PDF"
        if source_name not in failed_sources:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            try:
                response = requests.get(pdf_url, timeout=10)
                if response.status_code in [403, 422]:
                    print(f"✗ Skipped arXiv PDF for {paper_id}: HTTP {response.status_code}")
                else:
                    response.raise_for_status()
                    pdf_path = PDF_DIR / f"{arxiv_id}.pdf"
                    with open(pdf_path, 'wb') as f:
                        f.write(response.content)
                    print(f"✓ Downloaded PDF for {paper_id} from arXiv")
                    return str(pdf_path)
            except requests.exceptions.Timeout:
                print(f"✗ Timeout arXiv PDF for {paper_id}")
                failed_sources.add(source_name)
            except Exception as e:
                print(f"✗ Failed arXiv PDF for {paper_id}: {e}")
                failed_sources.add(source_name)
    
    # 2. OpenAlex oa_url (direct link to open access PDF)
    oa_url = paper.get("open_access", {}).get("oa_url")
    if oa_url:
        source_name = "OpenAlex OA URL"
        if source_name not in failed_sources:
            try:
                response = requests.get(oa_url, timeout=10)
                if response.status_code in [403, 422]:
                    print(f"✗ Skipped OpenAlex OA URL for {paper_id}: HTTP {response.status_code}")
                else:
                    response.raise_for_status()
                    filename = f"{paper_id.replace('/', '_')}.pdf"
                    pdf_path = PDF_DIR / filename
                    with open(pdf_path, 'wb') as f:
                        f.write(response.content)
                    print(f"✓ Downloaded PDF for {paper_id} from OpenAlex OA URL")
                    return str(pdf_path)
            except requests.exceptions.Timeout:
                print(f"✗ Timeout OpenAlex OA URL for {paper_id}")
                failed_sources.add(source_name)
            except Exception as e:
                print(f"✗ Failed OpenAlex OA URL for {paper_id}: {e}")
                failed_sources.add(source_name)
                if "aclweb.org" in oa_url and arxiv_id:
                    mirror_name = "arXiv mirror"
                    if mirror_name not in failed_sources:
                        print(f"⚠ ACL OA URL failed; trying arXiv mirror for {paper_id}")
                        try:
                            mirror_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                            mirror_response = requests.get(mirror_url, timeout=10)
                            if mirror_response.status_code in [403, 422]:
                                print(f"✗ Skipped arXiv mirror for {paper_id}: HTTP {mirror_response.status_code}")
                            else:
                                mirror_response.raise_for_status()
                                filename = f"{paper_id.replace('/', '_')}.pdf"
                                pdf_path = PDF_DIR / filename
                                with open(pdf_path, 'wb') as f:
                                    f.write(mirror_response.content)
                                print(f"✓ Downloaded PDF for {paper_id} from arXiv mirror")
                                return str(pdf_path)
                        except requests.exceptions.Timeout:
                            print(f"✗ Timeout arXiv mirror for {paper_id}")
                            failed_sources.add(mirror_name)
                        except Exception as mirror_error:
                            print(f"✗ Failed arXiv mirror for {paper_id}: {mirror_error}")
                            failed_sources.add(mirror_name)
    
    # 3. Unpaywall
    if doi:
        source_name = "Unpaywall"
        if source_name not in failed_sources:
            try:
                # Strip DOI prefix to get bare DOI
                bare_doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
                up_url = f"https://api.unpaywall.org/v2/{bare_doi}?email=researchpilot@example.com"
                response = requests.get(up_url, timeout=10)
                response.raise_for_status()
                data = response.json()
                if data.get("best_oa_location", {}).get("url_for_pdf"):
                    pdf_url = data["best_oa_location"]["url_for_pdf"]
                    pdf_response = requests.get(pdf_url, timeout=10)
                    if pdf_response.status_code in [403, 422]:
                        print(f"✗ Skipped Unpaywall PDF for {paper_id}: HTTP {pdf_response.status_code}")
                    else:
                        pdf_response.raise_for_status()
                        filename = f"{paper_id.replace('/', '_')}.pdf"
                        pdf_path = PDF_DIR / filename
                        with open(pdf_path, 'wb') as f:
                            f.write(pdf_response.content)
                        print(f"✓ Downloaded PDF for {paper_id} from Unpaywall")
                        return str(pdf_path)
                else:
                    print(f"✗ No PDF from Unpaywall for {paper_id}")
            except requests.exceptions.Timeout:
                print(f"✗ Timeout Unpaywall for {paper_id}")
                failed_sources.add(source_name)
            except Exception as e:
                print(f"✗ Failed Unpaywall for {paper_id}: {e}")
                failed_sources.add(source_name)

    print(f"✗ All sources failed for {paper_id}: {sorted(failed_sources)}")
    return None

def download_pdf(arxiv_id):
    # Legacy function, keep for compatibility
    if not arxiv_id:
        return None

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    pdf_path = PDF_DIR / f"{arxiv_id}.pdf"

    if pdf_path.exists():
        return str(pdf_path)

    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()

        with open(pdf_path, 'wb') as f:
            f.write(response.content)

        return str(pdf_path)
    except Exception as e:
        print(f"Failed to download PDF for {arxiv_id}: {e}")
        return None