from __future__ import annotations

from typing import List

from app.agents.base_agent import BaseAgent
from app.core.state import ResearchState
from app.services.pdf_downloader import download_pdf_with_fallbacks
from app.services.pdf_extractor import extract_text_from_pdf
from app.services.text_processor import process_documents


class AcquisitionAgent(BaseAgent):
    """Download PDFs, extract text, and prepare documents for chunking."""

    name = "AcquisitionAgent"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate paper text and create processed document chunks."""
        try:
            extracted_ids = set()
            for paper in state.papers:
                pdf_path = download_pdf_with_fallbacks(paper)
                if not pdf_path:
                    self._log(f"No PDF/text available for {paper.get('paper_id')}")
                    continue

                downloaded_arxiv_id = paper.get("arxiv_id")
                recorded_arxiv_id = state.metadata_store.get(paper.get("paper_id"), {}).get("arxiv_id")
                if downloaded_arxiv_id and recorded_arxiv_id and downloaded_arxiv_id != recorded_arxiv_id:
                    self._log(
                        f"arXiv ID mismatch for {paper.get('paper_id')}: downloaded {downloaded_arxiv_id} != recorded {recorded_arxiv_id}. Skipping."
                    )
                    continue

                full_text = extract_text_from_pdf(pdf_path)
                if len(full_text) > 500:
                    paper["full_text"] = full_text
                    extracted_ids.add(paper["paper_id"])
                    self._log(f"Extracted full text for {paper.get('paper_id')}")
                else:
                    self._log(f"Extracted text too short for {paper.get('paper_id')}")

            state.papers_with_extracted_text = extracted_ids
            state.filtered_papers = [paper for paper in state.papers if paper["paper_id"] in extracted_ids]

            if len(state.filtered_papers) < 2:
                self._log(f"Warning: Only {len(state.filtered_papers)} papers have extractable text. Results may be limited.")

            state.documents = process_documents(state.papers)
            state.diagnostics["chunk_count"] = len(state.documents)
            state.diagnostics["processed_documents"] = len(state.documents)
            self._log(f"Processed {len(state.documents)} document chunks")
        except Exception as error:
            error_message = f"Acquisition failed: {error}"
            self._log(error_message)
            state.errors.append(error_message)
        return state
