import json
import logging
import re
from typing import Dict, List, Any
import os
from dotenv import load_dotenv
import httpx

from app.services.llm_service import ollama_generate_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def generate_report(
    top_chunks: List[Dict],
    metadata: Dict,
    original_query: str,
    conversation_history: List[Dict],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Generate a structured research report using a single LLM call.

    Args:
        top_chunks: List of retrieved chunks with metadata
        metadata: Metadata store keyed by paper_id
        original_query: The original research query
        conversation_history: List of previous queries in the conversation
        verbose: Enable detailed logging

    Returns:
        Dict with keys: report, coverage_score
        report: Structured report with sections
        coverage_score: Number of unique papers cited
    """
    if verbose:
        logger.info("AGENT: Generating structured report")

    try:
        # Prepare context from top chunks
        context = _prepare_context(top_chunks, metadata, verbose)

        # Build conversation context
        conv_context = _prepare_conversation_context(conversation_history)

        # Generate report with single LLM call
        report_text = _call_llm_for_report(original_query, context, conv_context, verbose)

        if not report_text:
            logger.error("Ollama generate returned empty report text. Returning partial/empty result.")
            return {
                "report": "",
                "coverage_score": 0,
                "cited_ids": []
            }

        # Parse and validate report structure
        parsed_report = _parse_report_structure(report_text, verbose)

        # Validate citations for faithfulness
        parsed_report, valid_count, invalid_removed, integrity_score = _validate_report_citations(parsed_report, metadata, verbose)

        # Verify reference indices and append warnings if necessary
        parsed_report = _validate_citation_indices(parsed_report, metadata, verbose)

        # Extract structured cited paper IDs from the report text
        cited_paper_ids = _extract_cited_paper_ids(parsed_report, metadata, verbose)

        # Calculate coverage score
        coverage_score = _calculate_coverage_score(parsed_report, verbose)

        if verbose:
            logger.info(f"Report generated with coverage score: {coverage_score}")
            logger.info(f"Citation integrity: {valid_count} valid, {invalid_removed} invalid removed, score={integrity_score:.2f}")

        return {
            "report": parsed_report,
            "coverage_score": coverage_score,
            "valid_citation_count": valid_count,
            "invalid_citations_removed": invalid_removed,
            "citation_integrity_score": integrity_score,
            "cited_paper_ids": cited_paper_ids
        }

    except httpx.TimeoutException:
        logger.error("Ollama generate timed out after 120s. Returning partial/empty result.")
        return {
            "report": "",
            "coverage_score": 0,
            "cited_ids": []
        }
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}", exc_info=verbose)
        # Return fallback report
        fallback_report = _create_fallback_report(original_query)
        return {
            "report": fallback_report,
            "coverage_score": 0
        }


def _prepare_context(top_chunks: List[Dict], metadata: Dict, verbose: bool) -> str:
    """Prepare context string from top 5 chunks only, with truncated content."""
    context_parts = []
    max_chunks = min(5, len(top_chunks))  # Use only top 5 chunks
    max_content_length = 400  # Limit chunk content to 400 chars

    for i, chunk in enumerate(top_chunks[:max_chunks]):
        # Support both dict-like chunks and langchain Document objects
        if hasattr(chunk, "metadata"):
            chunk_metadata = getattr(chunk, "metadata", {}) or {}
            content = getattr(chunk, "page_content", "") or ""
        else:
            chunk_metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
            content = chunk.get("page_content", "") if isinstance(chunk, dict) else ""

        # Truncate content to reduce context size
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."

        paper_id = chunk_metadata.get("paper_id", "unknown")
        paper_meta = metadata.get(paper_id, {})
        title = paper_meta.get("title", "Unknown Title")
        authors = paper_meta.get("authors", [])
        author_string = _format_author_list(authors)

        chunk_text = f"[{title}] by {author_string}\n{content}"
        context_parts.append(chunk_text)

        if verbose:
            logger.debug(f"Report context chunk {i}: paper_id={paper_id}, title={title}")

    return "\n\n".join(context_parts)


def _prepare_conversation_context(conversation_history: List[Dict]) -> str:
    """Prepare conversation history context."""
    if not conversation_history:
        return "No previous conversation."

    history_parts = []
    for entry in conversation_history[-3:]:  # Last 3 entries
        query = entry.get("query", "")
        history_parts.append(f"Previous query: {query}")

    return "\n".join(history_parts)


def _format_author_list(authors: List[str]) -> str:
    """Format author metadata for report context."""
    if not authors:
        return "Unknown authors"
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]} and {authors[1]}"
    return f"{authors[0]}, {authors[1]}, et al."


def _normalize_title(title: str) -> str:
    if not title:
        return ""
    normalized = re.sub(r"[\W_]+", " ", title).strip().lower()
    return re.sub(r"\s+", " ", normalized)


def _extract_cited_paper_ids(report_text: str, metadata: Dict, verbose: bool) -> List[str]:
    """Extract OpenAlex paper IDs from citations in both numeric [1] and author-year formats."""
    cited_ids = []
    sources = list(metadata.values())  # Convert metadata dict to list for indexing

    # Pattern 1: numeric [1], [2] etc.
    numeric_indices = set(int(m) for m in re.findall(r'\[(\d+)\]', report_text))
    for idx in numeric_indices:
        if 1 <= idx <= len(sources):
            paper_id = sources[idx - 1].get('paper_id', '')
            if paper_id:
                cited_ids.append(paper_id)

    # Pattern 2: title in brackets [Title Here]
    title_matches = re.findall(r'\[([^\]]+)\]', report_text)
    for title in title_matches:
        normalized_title = _normalize_title(title)
        for source in sources:
            src_title = _normalize_title(source.get('title', ''))
            if normalized_title == src_title:
                paper_id = source.get('paper_id', '')
                if paper_id:
                    cited_ids.append(paper_id)
                break

    # Pattern 3: author-year fallback (Author et al., YYYY)
    if not cited_ids:
        year_matches = re.findall(r'\(([A-Z][a-z]+(?:\s+et\s+al\.)?),?\s+(\d{4})\)', report_text)
        for author, year in year_matches:
            for source in sources:
                src_authors = source.get('authors', [])
                src_year = str(source.get('year', ''))
                if src_year == year and any(author.lower() in a.lower() for a in src_authors):
                    paper_id = source.get('paper_id', '')
                    if paper_id:
                        cited_ids.append(paper_id)
                    break

    return list(set(filter(None, cited_ids)))

    if verbose:
        logger.info(f"Extracted {len(unique_ids)} cited OpenAlex IDs from report text")

    return unique_ids


def _validate_citation_indices(parsed_report: str, metadata: Dict, verbose: bool) -> str:
    """Check citation index references for out-of-range indices and append warnings."""
    sources = list(metadata.values())
    total_sources = len(sources)
    warnings = []

    for match in re.finditer(r"\[(\d+)\]", parsed_report):
        idx = int(match.group(1))
        if idx < 1 or idx > total_sources:
            warnings.append(f"Citation index [{idx}] is out of range for {total_sources} available sources.")

    if warnings:
        warning_section = "\n\n## Citation Warnings\n" + "\n".join(f"- {w}" for w in warnings)
        if warning_section not in parsed_report:
            parsed_report = parsed_report.strip() + warning_section
        if verbose:
            for warning in warnings:
                logger.warning(warning)

    return parsed_report


def _call_llm_for_report(query: str, context: str, conv_context: str, verbose: bool) -> str:
    """Make single LLM call to generate structured report (simplified for speed)."""
    # FIX 1: Add diagnostic logging for context injection
    context_chunks = context.split('\n\n') if context else []
    context_text = context
    prompt_length = len(context_text)
    logger.info(f"Report prompt context: {len(context_chunks)} chunks, {prompt_length} chars injected")
    if prompt_length < 500:
        logger.error("CRITICAL: Context injection is empty or near-empty. Retrieved chunks not reaching LLM prompt.")

    prompt = f"""You are a research assistant. Use ONLY the following research papers
to write a structured report. Cite papers by their source index number.

RETRIEVED CONTEXT:
{context_text}

QUERY: {query}

Write a structured report with: Summary, Key Methods (with inline citations [1], [2]...),
Comparison Table, Research Gaps, and References.
IMPORTANT: Only cite papers whose content appears in the RETRIEVED CONTEXT above.
Do not cite papers from memory or training data.

CITATION FORMAT: Use only numeric citations in square brackets [1], [2], [3].
Map each number to the source index in the RETRIEVED CONTEXT below.
Do NOT use author-year format. Do NOT invent citations not in the context.

## Comparison Table
A markdown table comparing methods across: Strengths, Limitations, Applicable Fraud Type, Citation.
Include at least 3 rows. Use only methods mentioned in the RETRIEVED CONTEXT.

Begin each Key Method description with a direct quote or paraphrase from the context."""

    try:
        report_text = ollama_generate_text(prompt)
        # FIX 5: Validate comparison table presence
        if "| Method" not in report_text and "| ---" not in report_text:
            logger.warning("Comparison table missing from generated report. LLM may have skipped it.")
        return report_text
    except httpx.TimeoutException:
        logger.error("Ollama generate timed out after 120s. Returning partial/empty result.")
        return ""
    except Exception as e:
        logger.error(f"LLM call failed: {str(e)}")
        raise


def _parse_report_structure(report_text: str, verbose: bool) -> str:
    """Parse and validate report structure, add missing sections if needed."""
    sections = ["## Summary", "## Key Methods", "## Research Gaps", "## References"]
    parsed_sections = {}

    # Split report into sections
    lines = report_text.split('\n')
    current_section = None
    current_content = []

    for line in lines:
        # Check if this is a section header
        if line.strip().startswith('## '):
            # Save previous section
            if current_section:
                parsed_sections[current_section] = '\n'.join(current_content).strip()

            # Start new section
            current_section = line.strip()
            current_content = []
        elif current_section:
            current_content.append(line)

    # Save last section
    if current_section:
        parsed_sections[current_section] = '\n'.join(current_content).strip()

    # Build complete report, adding missing sections
    complete_report = []
    for section in sections:
        if section in parsed_sections and parsed_sections[section]:
            complete_report.append(f"{section}\n{parsed_sections[section]}")
        else:
            # Add fallback content for missing sections
            fallback_content = _get_fallback_content(section)
            complete_report.append(f"{section}\n{fallback_content}")

    return '\n\n'.join(complete_report)


def _get_fallback_content(section: str) -> str:
    """Provide fallback content for missing report sections."""
    fallbacks = {
        "## Summary": "Summary not available from current context.",
        "## Key Methods": "- Methods not clearly identified in retrieved papers.",
        "## Research Gaps": "- Research gaps not identified in current context.",
        "## References": "- No references available."
    }
    return fallbacks.get(section, "Content not available.")


def _validate_report_citations(parsed_report: str, metadata: Dict, verbose: bool) -> (str, int, int, float):
    """Validate citations and remove invalid ones safely from the generated report."""
    if "## References" not in parsed_report:
        return parsed_report, 0, 0, 1.0

    header, references = parsed_report.split("## References", 1)
    valid_titles = set()
    for line in references.splitlines():
        match = re.search(r'\[([^\]]+)\]', line)
        if match:
            valid_titles.add(match.group(1).strip().lower())

    valid_count = 0
    invalid_count = 0

    def _replace_citation(match):
        nonlocal valid_count, invalid_count
        title = match.group(1).strip()
        if title.lower() in valid_titles:
            valid_count += 1
            return match.group(0)
        invalid_count += 1
        return "[Citation removed - source not available]"

    sanitized_header = re.sub(r'\[([^\]]+)\]', _replace_citation, header)
    sanitized_report = f"{sanitized_header}## References{references}"
    total = valid_count + invalid_count
    integrity_score = (valid_count / total) if total > 0 else 1.0

    if verbose:
        logger.info(f"Citation validation: {valid_count} valid, {invalid_count} invalid, score={integrity_score:.2f}")

    return sanitized_report, valid_count, invalid_count, integrity_score


def _calculate_coverage_score(parsed_report: str, verbose: bool) -> int:
    """Calculate coverage score based on unique papers cited."""
    citations = re.findall(r'\[([^\]]+)\]', parsed_report)

    unique_papers = set()
    for citation in citations:
        clean_citation = citation.strip('[]')
        if clean_citation and len(clean_citation) > 10:
            unique_papers.add(clean_citation.lower())

    coverage_score = len(unique_papers)

    if verbose:
        logger.info(f"Coverage score: {coverage_score} unique papers cited")

    return coverage_score


def _create_fallback_report(query: str) -> str:
    """Create a minimal fallback report when generation fails."""
    return f"""## Summary
Unable to generate summary for query: {query}

## Key Methods
- Methods could not be extracted from available context.

## Comparison Table
| Method | Strengths | Limitations | Citation |
|--------|-----------|-------------|----------|
| N/A | N/A | N/A | N/A |

## Research Gaps
- Research gaps could not be identified.

## References
- No references available due to processing error."""