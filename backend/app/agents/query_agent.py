import json
import logging
from typing import Dict
import os
from dotenv import load_dotenv
import httpx

from app.services.llm_service import ollama_generate_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _normalize_query(query: str, verbose: bool = False) -> str:
    """
    Normalize project-oriented or implementation queries into research-friendly queries.
    """
    query_lower = query.lower()
    
    # Detect project-oriented patterns
    project_patterns = [
        "project ideas", "project idea", "projects for", "implement", "implementation",
        "build a", "create a", "develop a", "how to", "tutorial", "guide",
        "step by step", "example of", "code for"
    ]
    
    is_project_query = any(pattern in query_lower for pattern in project_patterns)
    
    if not is_project_query:
        return query
    
    # Transform to research query
    transformations = {
        "project ideas for": "research on",
        "project idea for": "research on",
        "projects for": "research on",
        "implement": "research on",
        "implementation": "research on",
        "build a": "research on",
        "create a": "research on",
        "develop a": "research on",
        "how to": "research on",
        "tutorial": "research on",
        "guide": "research on",
        "step by step": "research on",
        "example of": "research on",
        "code for": "research on"
    }
    
    normalized = query
    for old, new in transformations.items():
        normalized = normalized.replace(old, new)
    
    # Add research context
    if "machine learning" in normalized.lower() or "ml" in normalized.lower():
        if "fraud detection" in normalized.lower():
            normalized = "machine learning fraud detection applications research"
        elif "system" in normalized.lower():
            normalized = "machine learning system architectures research"
    
    if verbose:
        logger.info(f"Normalized project query '{query}' to research query '{normalized}'")
    
    return normalized


def process_query(query: str, verbose: bool = False) -> Dict[str, any]:
    """
    Decompose a query into primary_query, sub_queries, and keywords.
    
    Args:
        query: Raw query string
        verbose: Enable detailed logging
    
    Returns:
        Dict with keys: primary_query, sub_queries, keywords
        Example: {
            "primary_query": "machine learning in drug discovery",
            "sub_queries": [
                "deep learning applications in pharmaceutical research",
                "AI models for protein structure prediction",
                "neural networks in drug toxicity assessment"
            ],
            "keywords": ["neural networks", "drug discovery", "machine learning", "proteins", "toxicity"]
        }
    """
    if not query or not query.strip():
        logger.warning("AGENT: Empty query provided")
        return {
            "primary_query": query or "",
            "sub_queries": [],
            "keywords": []
        }
    
    # Normalize query for better academic retrieval
    normalized_query = _normalize_query(query, verbose)
    
    if verbose:
        logger.info(f"AGENT: Processing query: {query}")
        if normalized_query != query:
            logger.info(f"AGENT: Normalized to: {normalized_query}")
    
    # Try to get LLM response with retry logic
    response = _call_llm_with_retry(normalized_query, verbose)
    
    # Try to parse JSON response
    try:
        parsed = _parse_query_response(response, query, verbose)
    except Exception as e:
        logger.error(f"Failed to parse query response: {str(e)}")
        parsed = _create_fallback_query(query)
    
    # Log results
    if verbose:
        logger.info(f"Query decomposed: {len(parsed['sub_queries'])} sub_queries, {len(parsed['keywords'])} keywords")
    else:
        logger.info(f"Query processed: {len(parsed['sub_queries'])} sub-queries, {len(parsed['keywords'])} keywords")
    
    return parsed


def _call_llm_with_retry(query: str, verbose: bool, max_retries: int = 2) -> str | None:
    """
    Call LLM to decompose query with retry logic.
    
    Args:
        query: Query to decompose
        verbose: Enable detailed logging
        max_retries: Number of retries on failure
    
    Returns:
        LLM response string, or None if all retries exhausted
    """
    prompt = f"""You are a research query decomposition assistant.

RESEARCH QUERY: "{query}"

Your task: Break down this research query into:
1. A primary_query (the main research topic)
2. 2-4 sub_queries (specific aspects to search for)
3. 3-5 keywords (key terms for the topic)

Return ONLY valid JSON in this exact format:
{{
    "primary_query": "...",
    "sub_queries": ["...", "...", ...],
    "keywords": ["...", "...", ...]
}}

Do not add any explanation or text outside the JSON."""

    for attempt in range(max_retries + 1):
        try:
            if verbose and attempt > 0:
                logger.info(f"Retry attempt {attempt}/{max_retries}")
            
            response_text = ollama_generate_text(prompt)
            if response_text:
                return response_text
            if verbose:
                logger.warning(f"LLM returned empty response on attempt {attempt + 1}")
            
            if attempt < max_retries:
                continue
            else:
                logger.warning("All LLM retries exhausted, using fallback")
                return None
        except httpx.TimeoutException:
            logger.error("Ollama generate timed out after 120s during query decomposition")
            return None
        except Exception as e:
            if verbose:
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries:
                continue
            else:
                logger.warning("All LLM retries exhausted, using fallback")
                return None
    
    return None


def _create_fallback_query(query: str) -> Dict[str, any]:
    """Create fallback query structure when parsing fails."""
    return {
        "primary_query": query,
        "sub_queries": [query],
        "keywords": []
    }


def _parse_query_response(response: str, original_query: str, verbose: bool) -> Dict[str, any]:
    """
    Parse LLM response into structured format with fallback.
    
    Args:
        response: LLM response string
        original_query: Original query for fallback
        verbose: Enable detailed logging
    
    Returns:
        Parsed dict with primary_query, sub_queries, keywords
    """
    # If no response, use fallback
    if not response:
        if verbose:
            logger.info("No LLM response, using fallback")
        return _create_fallback_query(original_query)
    
    # Try to extract JSON from response (may have extra text)
    try:
        # Try direct JSON parse first
        parsed = json.loads(response.strip())
    except json.JSONDecodeError:
        # Try to extract JSON block from response
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                parsed = json.loads(json_str)
            else:
                raise ValueError("No JSON block found")
        except Exception as e:
            if verbose:
                logger.warning(f"JSON parsing failed: {str(e)}, using fallback")
            return _create_fallback_query(original_query)
    
    # Validate and normalize parsed response
    try:
        result = {
            "primary_query": str(parsed.get("primary_query", original_query)).strip(),
            "sub_queries": [str(q).strip() for q in parsed.get("sub_queries", [original_query]) if q],
            "keywords": [str(k).strip() for k in parsed.get("keywords", []) if k]
        }
        
        # Ensure minimum structure
        if not result["primary_query"]:
            result["primary_query"] = original_query
        if not result["sub_queries"]:
            result["sub_queries"] = [original_query]
        
        return result
    except Exception as e:
        if verbose:
            logger.warning(f"Response validation failed: {str(e)}, using fallback")
        return _create_fallback_query(original_query)