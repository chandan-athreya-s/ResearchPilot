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
    
    @staticmethod
    def detect_query_type(query: str) -> str:
        """Detect the primary intent type of the query.
        
        Enhanced heuristics:
        - "<topic> for <domain>" patterns → survey/research overview
        - Better comparison keyword detection
        - More nuanced scoring based on context
        
        Args:
            query: The research query string
            
        Returns:
            One of: "comparison", "survey", "implementation", "challenges", "general"
        """
        query_lower = query.lower()
        
        # Initialize scores
        type_scores = {
            "comparison": 0,
            "survey": 0,
            "implementation": 0,
            "challenges": 0
        }
        
        # Count keyword matches for each type
        for keyword in QueryAnalyzer.COMPARISON_KEYWORDS:
            if keyword in query_lower:
                type_scores["comparison"] += 1
                
        for keyword in QueryAnalyzer.SURVEY_KEYWORDS:
            if keyword in query_lower:
                type_scores["survey"] += 1
                
        for keyword in QueryAnalyzer.IMPLEMENTATION_KEYWORDS:
            if keyword in query_lower:
                type_scores["implementation"] += 1
                
        for keyword in QueryAnalyzer.CHALLENGES_KEYWORDS:
            if keyword in query_lower:
                type_scores["challenges"] += 1
        
        # ENHANCED HEURISTICS
        
        # 1. "<topic> for <domain>" patterns → survey/research overview
        # Examples: "reinforcement learning for robotics", "machine learning for healthcare"
        # Must have complete pattern: word + for/in/applied to/within + word
        # AND avoid triggering on questions like "why does X occur in Y"
        for_pattern = re.search(r'\b\w+\s+(?:for|in|applied\s+to|within)\s+\w+', query_lower)
        if for_pattern and len(for_pattern.group().split()) >= 3:  # Ensure complete pattern
            # Additional check: avoid triggering on questions (why/what/how/when/where at start)
            if not query_lower.startswith(('why ', 'what ', 'how ', 'when ', 'where ')):
                type_scores["survey"] += 2  # Strong boost for survey behavior
        
        # 2. Comparison patterns with "vs", "versus", or "compared to"
        if re.search(r'\b(?:vs|versus|compared\s+to|vs\.)\b', query_lower):
            type_scores["comparison"] += 2
        
        # 3. Trade-off analysis patterns
        if re.search(r'\b(?:trade.?off|trade.?offs|pros\s+and\s+cons|advantages?\s+and\s+disadvantages?)\b', query_lower):
            type_scores["comparison"] += 2
        
        # 4. Implementation-focused question starters
        if query_lower.startswith(('how to', 'how do', 'how can', 'implementing', 'building')):
            type_scores["implementation"] += 2
        
        # 5. Challenge/problem-focused question starters
        # Only boost if followed by actual challenge/problem keywords
        if query_lower.startswith(('what are the', 'what is the', 'why does', 'why do')):
            challenge_followers = ['problem', 'challenge', 'issue', 'limitation', 'difficulty', 'barrier', 'error', 'failure', 'occur', 'happen', 'cause']
            if any(word in query_lower for word in challenge_followers):
                type_scores["challenges"] += 2
        
        # 6. Survey patterns with broad scope indicators
        broad_scope_words = ['overview', 'landscape', 'state', 'current', 'recent', 'advances', 'developments', 'trends']
        if any(word in query_lower for word in broad_scope_words):
            type_scores["survey"] += 1
        
        # 7. Implementation patterns with practical indicators
        practical_words = ['practical', 'real-world', 'production', 'deploy', 'scale', 'efficient']
        if any(word in query_lower for word in practical_words):
            type_scores["implementation"] += 1
        
        # Return the type with highest score, default to "general"
        if max(type_scores.values()) > 0:
            return max(type_scores, key=type_scores.get)
        return "general"
    
    @staticmethod
    def extract_focus_terms(query: str, max_terms: int = 5) -> List[str]:
        """Extract important focus terms (2+ word noun phrases) from query.
        
        Strategy:
        1. Split into tokens and normalize
        2. Find noun phrases (sequences of non-stop words)
        3. Prefer multi-word phrases over single words
        4. Return top N by frequency/position
        
        Args:
            query: The research query string
            max_terms: Maximum number of focus terms to extract (default: 5)
            
        Returns:
            List of extracted focus terms, ordered by importance
        """
        # Normalize and tokenize
        query_lower = query.lower()
        # Remove punctuation but keep spaces
        query_clean = re.sub(r'[^\w\s]', ' ', query_lower)
        tokens = query_clean.split()
        
        # Find noun phrases (consecutive non-stop words)
        phrases = []
        current_phrase = []
        
        for token in tokens:
            if token not in QueryAnalyzer.STOP_WORDS and len(token) > 2:
                current_phrase.append(token)
            else:
                if len(current_phrase) >= 2:
                    # Multi-word phrase
                    phrase = " ".join(current_phrase)
                    phrases.append(phrase)
                elif len(current_phrase) == 1:
                    # Single word - keep for fallback but lower priority
                    phrases.append(current_phrase[0])
                current_phrase = []
        
        # Don't forget last phrase
        if len(current_phrase) >= 2:
            phrase = " ".join(current_phrase)
            phrases.append(phrase)
        elif len(current_phrase) == 1:
            phrases.append(current_phrase[0])
        
        # Remove duplicates while preserving order (prioritize first occurrence)
        seen = set()
        unique_phrases = []
        for phrase in phrases:
            if phrase not in seen:
                seen.add(phrase)
                unique_phrases.append(phrase)
        
        # Prioritize multi-word phrases, then return top N
        multi_word = [p for p in unique_phrases if " " in p]
        single_word = [p for p in unique_phrases if " " not in p]
        
        # Return multi-word first, fill with single words if needed
        result = multi_word[:max_terms]
        if len(result) < max_terms:
            result.extend(single_word[:max_terms - len(result)])
        
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


def analyze_query(query: str) -> Dict:
    """Convenience function for query analysis.
    
    Args:
        query: The research query string
        
    Returns:
        Dictionary with query_type, focus_terms, and original_query
    """
    return QueryAnalyzer.analyze(query)
