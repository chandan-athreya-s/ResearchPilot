"""Lightweight query type detection and focus term extraction.

Detects query intent types and extracts important focus terms to inform retrieval behavior.
No external dependencies beyond standard library and existing packages.
"""

import re
from typing import Dict, List, Tuple


class QueryAnalyzer:
    """Analyzes queries to determine type and extract focus terms."""
    
    # Query type patterns - keyword-based detection
    COMPARISON_KEYWORDS = {
        "comparison", "compare", "versus", "vs", "difference", "differ", 
        "distinguish", "distinguish between", "contrast", "alternative",
        "comparative", "compared to", "compared with", "vs.", "trade-off",
        "trade-offs", "pros and cons", "advantages and disadvantages"
    }
    
    SURVEY_KEYWORDS = {
        "survey", "review", "overview", "state of the art", "sota",
        "landscape", "systematic", "systematic review", "literature",
        "comprehensive", "exhaustive", "taxonomy", "categorization",
        "classification", "survey of", "review of"
    }
    
    IMPLEMENTATION_KEYWORDS = {
        "implement", "implementation", "build", "develop", "application",
        "apply", "practical", "engineer", "architecture", "system",
        "framework", "toolkit", "library", "tool", "deploy", "deployment",
        "production", "how to", "methodology", "method"
    }
    
    CHALLENGES_KEYWORDS = {
        "challenge", "limitation", "problem", "issue", "difficulty",
        "obstacle", "barrier", "constraint", "limitation", "weakness",
        "gap", "limitation", "fail", "failure", "error", "bug",
        "open problem", "unsolved", "unresolved", "future work"
    }
    
    # Common stop words for focus term extraction
    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "by", "from", "with", "is", "are", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "can", "as", "if", "so", "than",
        "this", "that", "these", "those", "i", "you", "he", "she", "it",
        "we", "they", "what", "which", "who", "when", "where", "why", "how"
    }

    NORMALIZATION_MAP = {
        "rag": "retrieval augmented generation",
        "retrieval augmented generation": "retrieval augmented generation",
        "fine tuning": "fine tuning",
        "fine-tuning": "fine tuning",
        "fine tune": "fine tuning",
        "llm": "large language model",
        "large language model": "large language model",
        "large language models": "large language model",
        "agentic workflows": "agentic workflows",
        "agentic workflow": "agentic workflows",
        "autonomous agents": "agentic workflows",
        "enterprise knowledge systems": "enterprise knowledge systems",
        "enterprise ai systems": "enterprise knowledge systems",
        "knowledge management systems": "enterprise knowledge systems",
    }
    
    @staticmethod
    def _matches_any_pattern(query: str, patterns: List[str]) -> bool:
        return any(re.search(pattern, query) for pattern in patterns)

    @staticmethod
    def _is_comparison_query(query: str) -> bool:
        comparison_patterns = [
            r"\bcomparison(?:\s+of|\s+between)?\b",
            r"\bcompare\b",
            r"\bvs\b",
            r"\bversus\b",
            r"\bdifferences?\s+between\b",
            r"\bcompared\s+(?:to|with)\b",
        ]
        return QueryAnalyzer._matches_any_pattern(query, comparison_patterns)

    @staticmethod
    def _is_implementation_query(query: str) -> bool:
        return any(keyword in query for keyword in QueryAnalyzer.IMPLEMENTATION_KEYWORDS)

    @staticmethod
    def _is_challenges_query(query: str) -> bool:
        return any(keyword in query for keyword in QueryAnalyzer.CHALLENGES_KEYWORDS)

    @staticmethod
    def _is_survey_query(query: str) -> bool:
        survey_patterns = [
            r"\bsurvey\b",
            r"\breview\b",
            r"\boverview\b",
            r"\bstate of the art\b",
            r"\blandscape\b",
            r"\bsystematic\b",
            r"\bcomprehensive\b",
            r"\btaxonomy\b",
            r"\bclassification\b",
        ]
        if QueryAnalyzer._matches_any_pattern(query, survey_patterns):
            return True

        if re.search(r"\b(recent|latest)\s+papers?\b", query):
            return False

        broader_survey_terms = [
            'overview', 'landscape', 'state', 'advances',
            'developments', 'trends', 'literature'
        ]
        if any(term in query for term in broader_survey_terms):
            if not QueryAnalyzer._matches_any_pattern(query, list(QueryAnalyzer.CHALLENGES_KEYWORDS)):
                return True

        return False

    @staticmethod
    def detect_query_type(query: str) -> str:
        """Detect the primary intent type of the query.

        Priority order:
        1. comparison
        2. implementation
        3. challenges
        4. survey
        5. general
        """
        query_lower = query.lower()

        if QueryAnalyzer._is_comparison_query(query_lower):
            return "comparison"
        if QueryAnalyzer._is_survey_query(query_lower):
            return "survey"
        if QueryAnalyzer._is_implementation_query(query_lower):
            return "implementation"
        if QueryAnalyzer._is_challenges_query(query_lower):
            return "challenges"
        return "general"
    
    @staticmethod
    def _normalize_entity(entity: str) -> str:
        entity = entity.strip().lower()
        entity = re.sub(r'\s+', ' ', entity)
        if entity in QueryAnalyzer.NORMALIZATION_MAP:
            return QueryAnalyzer.NORMALIZATION_MAP[entity]
        return entity

    def _split_known_aliases(entity: str) -> List[str]:
        entity = entity.strip().lower()
        found = []
        remaining = entity
        for alias in sorted(QueryAnalyzer.NORMALIZATION_MAP.keys(), key=len, reverse=True):
            pattern = rf"\b{re.escape(alias)}\b"
            while re.search(pattern, remaining):
                found.append(QueryAnalyzer.NORMALIZATION_MAP[alias])
                remaining = re.sub(pattern, ' ', remaining, count=1)
        remaining = re.sub(r'\s+', ' ', remaining).strip()
        if remaining and len(remaining.split()) > 1:
            found.append(remaining)
        return found if found else [entity]

    def extract_focus_terms(query: str, max_terms: int = 5) -> List[str]:
        """Extract important focus terms and normalized entities from query."""
        query_lower = query.lower()
        query_clean = re.sub(r'[\-\/]', ' ', query_lower)
        query_clean = re.sub(r'[^\w\s]', ' ', query_clean)
        query_clean = re.sub(r'\s+', ' ', query_clean).strip()

        # Capture trailing context phrases from "for" / "in" / "using"
        tail_match = re.search(r'\b(?:for|in|within|using|across)\s+(.+)$', query_clean)
        tail_phrase = tail_match.group(1).strip() if tail_match else ""

        # Split into meaningful segments and conjunctive chunks
        separators = re.split(r"\s*(?:,|;|\band\b|\bor\b|\bversus\b|\bvs\.?\b|\bwith\b|\bfor\b|\bin\b|\busing\b|\bwithin\b)\s*", query_clean)
        phrases = []
        for segment in separators:
            segment = segment.strip()
            if not segment:
                continue
            cleaned = re.sub(r'\b(?:the|a|an|of|to|for|in|with|using|on|at|by|from|within|this|that|these|those)\b', ' ', segment)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            if not cleaned:
                continue
            phrases.append(cleaned)

        if tail_phrase and tail_phrase not in phrases:
            phrases.append(tail_phrase)

        normalized_phrases = []
        seen = set()
        for phrase in phrases:
            expanded = QueryAnalyzer._split_known_aliases(phrase)
            for normalized in expanded:
                normalized = QueryAnalyzer._normalize_entity(normalized)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    normalized_phrases.append(normalized)

        multi_word = [p for p in normalized_phrases if ' ' in p]
        single_word = [p for p in normalized_phrases if ' ' not in p]

        result = []
        for phrase in multi_word:
            if len(result) < max_terms:
                result.append(phrase)
        for phrase in single_word:
            if len(result) < max_terms:
                result.append(phrase)

        return result[:max_terms]
    
    @staticmethod
    def analyze(query: str) -> Dict:
        """Analyze a query and return comprehensive intent information.
        
        Returns:
            Dictionary with:
            - query_type: str (comparison, survey, implementation, challenges, general)
            - focus_terms: List[str] (extracted important terms)
            - original_query: str (the input query)
        """
        return {
            "query_type": QueryAnalyzer.detect_query_type(query),
            "focus_terms": QueryAnalyzer.extract_focus_terms(query),
            "original_query": query
        }


def extract_comparison_pairs(query: str) -> List[Tuple[str, str]]:
    """Extract comparison element pairs from comparison queries.
    
    Identifies what is being compared (A vs B).
    
    Args:
        query: The research query string
        
    Returns:
        List of (item_a, item_b) tuples being compared
    """
    query_lower = query.lower().strip()
    query_lower = re.sub(r"[\-—]", " ", query_lower)
    
    # Comprehensive comparison patterns
    patterns = [
        (r"comparison\s+(?:between|of)\s+(.+?)\s+and\s+(.+?)(?:\.|$|:|;|\?)", 2),
        (r"compare\s+(.+?)\s+(?:and|with|to|versus|vs\.?)\s+(.+?)(?:\.|$|:|;|\?)", 2),
        (r"(.+?)\s+(?:vs|versus|v\.)\s+(.+?)(?:\.|$|:|;|\?|in|for)", 2),
        (r"(.+?)\s+compared\s+(?:to|with)\s+(.+?)(?:\.|$|:|;|\?)", 2),
        (r"(.+?)\s+against\s+(.+?)(?:\.|$|:|;|\?)", 2),
        (r"difference(?:s)?\s+between\s+(.+?)\s+and\s+(.+?)(?:\.|$|:|;|\?)", 2),
        (r"contrast(?:ing|ing)?\s+(.+?)\s+(?:and|with)\s+(.+?)(?:\.|$|:|;|\?)", 2),
    ]
    
    pairs = []
    for pattern, group_count in patterns:
        matches = re.finditer(pattern, query_lower)
        for match in matches:
            if group_count == 2:
                left = match.group(1).strip()
                right = match.group(2).strip()
            else:
                left = match.group(1).strip()
                right = match.group(2).strip()
            
            # Clean extracted terms
            left = _normalize_comparison_side(left)
            right = _normalize_comparison_side(right)
            
            if left and right and len(left) > 2 and len(right) > 2:
                pairs.append((left, right))
    
    return pairs


def _normalize_comparison_side(side: str) -> str:
    side = side.strip()
    side = re.sub(r'^(the|a|an)\s+', '', side)
    side = re.sub(r'\s+(?:for|in|on|to|using|with)\s+.*$', '', side)
    side = re.sub(r'[,:;.?]+.*$', '', side)
    return side.strip()


def analyze_query(query: str) -> Dict:
    """Convenience function for query analysis.
    
    Args:
        query: The research query string
        
    Returns:
        Dictionary with query_type, focus_terms, comparison_pairs, and original_query
    """
    analysis = QueryAnalyzer.analyze(query)
    
    # Add comparison pairs for comparison queries
    if analysis["query_type"] == "comparison":
        analysis["comparison_pairs"] = extract_comparison_pairs(query)
    else:
        analysis["comparison_pairs"] = []
    
    return analysis
