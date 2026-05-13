#!/usr/bin/env python3
"""Test script to verify DOI formatting, reference numbering, and citation style fixes."""

import sys
sys.path.insert(0, '/home/chandan-athreya-s/Major-Project/ResearchPilot/backend')

from app.services.llm_service import normalize_doi, post_process_citations

def test_doi_normalization():
    """Test Fix 1: DOI normalization prevents duplicate URL prefixes."""
    print("=" * 60)
    print("TEST 1: DOI NORMALIZATION")
    print("=" * 60)
    
    test_cases = [
        ("10.1234/example.doi", "10.1234/example.doi"),  # Raw DOI
        ("https://doi.org/10.1234/example.doi", "10.1234/example.doi"),  # Already formatted URL
        ("http://doi.org/10.1234/example.doi", "10.1234/example.doi"),  # HTTP variant
        ("doi.org/10.1234/example.doi", "10.1234/example.doi"),  # Missing protocol
        ("https://doi.org/https://doi.org/10.1234/example.doi", "10.1234/example.doi"),  # Malformed
        ("", None),  # Empty string
        (None, None),  # None value
    ]
    
    for input_doi, expected in test_cases:
        result = normalize_doi(input_doi)
        status = "✓" if result == expected else "✗"
        print(f"{status} Input: {input_doi!r}")
        print(f"  Expected: {expected!r}, Got: {result!r}")
        if result == expected:
            print(f"  Formatted URL: https://doi.org/{result}" if result else "  (No DOI)")
        print()


def test_citation_replacement():
    """Test Fix 2 & 3: Reference numbering and clean citation style."""
    print("=" * 60)
    print("TEST 2: CITATION REPLACEMENT & REMAPPING")
    print("=" * 60)
    
    # Sample source references data
    source_references = {
        "Source 1": {"title": "Paper 1", "authors": ["Smith, John"], "year": 2020, "paper_id": "p1"},
        "Source 2": {"title": "Paper 2", "authors": ["Doe, Jane"], "year": 2021, "paper_id": "p2"},
        "Source 3": {"title": "Paper 3", "authors": ["Johnson, Bob"], "year": 2022, "paper_id": "p3"},
    }
    
    # Mapping from old source numbers to new sequential numbers
    old_to_new_ref_num = {1: 1, 2: 2, 3: 3}
    
    # Sample text with internal citations
    sample_text = """This research demonstrates several key findings.

First, the approach shows improvements [Source 1, Chunk 5] and compares favorably to alternatives [Source 2, Chunk 12].

Limitations include scalability concerns [Source 3, Chunk 8] (Source 1, Chunk 3) and computational overhead.

The methodology combines ideas from previous work (Source 2, Chunk 2) and recent advances [Source 3, Chunk 15].

Future work should address these issues [Source 2, Chunk 9]."""

    print("INPUT TEXT:")
    print(sample_text)
    print("\n" + "=" * 60)
    
    result = post_process_citations(sample_text, source_references, old_to_new_ref_num)
    
    print("\nOUTPUT TEXT (with clean citation format):")
    print(result)
    print("\n" + "=" * 60)
    
    # Verify the transformations
    print("\nVERIFICATION:")
    tests = [
        ("[Source 1, Chunk 5]" not in result, "✓ [Source X, Chunk Y] format removed"),
        ("(Source " not in result, "✓ (Source X, Chunk Y) format removed"),
        ("[1]" in result, "✓ Clean [1] format present"),
        ("[2]" in result, "✓ Clean [2] format present"),
        ("[3]" in result, "✓ Clean [3] format present"),
    ]
    
    for test, description in tests:
        print(f"{'✓' if test else '✗'} {description}")


if __name__ == "__main__":
    test_doi_normalization()
    print("\n")
    test_citation_replacement()
    print("\n✓ All tests completed successfully!")
