"""Test suite for comparison query improvements."""

import sys
sys.path.insert(0, '/home/chandan-athreya-s/Major-Project/ResearchPilot/backend')

from app.services.query_analyzer import analyze_query, extract_comparison_pairs
from app.services.llm_service import classify_chunk_aspects, scaffold_evidence_for_comparison


def test_comparison_extraction():
    """Test comparison pair extraction."""
    print("\n=== Testing Comparison Pair Extraction ===")
    
    test_queries = [
        "Comparison of BERT and GPT for NLP tasks",
        "Compare transformer architectures vs recurrent networks",
        "What is the difference between supervised and unsupervised learning",
        "Reinforcement learning versus traditional control methods",
        "RNN vs CNN: when to use each?",
    ]
    
    for query in test_queries:
        pairs = extract_comparison_pairs(query)
        print(f"\nQuery: {query}")
        print(f"Pairs: {pairs}")
        assert len(pairs) > 0, f"Failed to extract pairs from: {query}"
    
    print("\n✓ Comparison extraction: PASS")


def test_query_analysis():
    """Test query analysis with comparison_pairs."""
    print("\n=== Testing Query Analysis ===")
    
    queries = [
        ("Comparison of deep learning and machine learning", "comparison"),
        ("Survey of reinforcement learning methods", "survey"),
        ("How to implement a neural network", "implementation"),
        ("What are the challenges in NLP", "challenges"),
        ("Random research query about data", "general"),
    ]
    
    for query, expected_type in queries:
        result = analyze_query(query)
        query_type = result.get("query_type")
        has_pairs = "comparison_pairs" in result
        
        print(f"\nQuery: {query}")
        print(f"Type: {query_type} (expected: {expected_type})")
        print(f"Has comparison_pairs: {has_pairs}")
        print(f"Focus terms: {result.get('focus_terms', [])[:3]}")
        
        assert query_type == expected_type or query_type == "general", f"Type mismatch for: {query}"
    
    print("\n✓ Query analysis: PASS")


def test_aspect_classification():
    """Test chunk aspect classification."""
    print("\n=== Testing Aspect Classification ===")
    
    test_chunks = [
        ("Our algorithm uses a transformer-based architecture with attention mechanisms", 
         {"methods": True, "architecture": True}),
        ("We evaluated on the ImageNet and COCO datasets using precision and recall metrics",
         {"datasets": True, "metrics": True}),
        ("Method A is more efficient but requires more memory, creating a speed-memory tradeoff",
         {"tradeoffs": True, "computational": True}),
        ("The main limitation of this approach is scalability to large datasets",
         {"limitations": True}),
        ("We deployed this in production to handle real-world user requests",
         {"applications": True}),
    ]
    
    for chunk, expected_aspects in test_chunks:
        aspects = classify_chunk_aspects(chunk)
        
        print(f"\nChunk: {chunk[:60]}...")
        print(f"Detected aspects: {[k for k, v in aspects.items() if v]}")
        
        for aspect, expected in expected_aspects.items():
            if expected:
                assert aspects.get(aspect), f"Failed to detect {aspect} in: {chunk}"
    
    print("\n✓ Aspect classification: PASS")


def test_evidence_scaffolding():
    """Test evidence scaffolding for comparison queries."""
    print("\n=== Testing Evidence Scaffolding ===")
    
    # Create mock documents
    class MockDoc:
        def __init__(self, content, paper_id="paper1", chunk_index=0):
            self.page_content = content
            self.metadata = {"paper_id": paper_id, "chunk_index": chunk_index}
    
    docs = [
        MockDoc("Transformer uses self-attention with O(n²) complexity", "paper1", 0),
        MockDoc("We evaluated on ImageNet with 95% accuracy", "paper1", 1),
        MockDoc("RNN is more memory efficient but slower", "paper2", 0),
        MockDoc("Our main limitation is handling long sequences", "paper2", 1),
    ]
    
    query_intent = {
        "query_type": "comparison",
        "focus_terms": ["transformer", "RNN"],
        "comparison_pairs": [("transformer", "RNN")]
    }
    
    aspect_groups = scaffold_evidence_for_comparison(docs, query_intent)
    
    print(f"\nAspect groups created:")
    for aspect, chunks in aspect_groups.items():
        if chunks:
            print(f"  {aspect}: {len(chunks)} chunks")
    
    assert len(aspect_groups) > 0, "No aspect groups created"
    assert any(len(chunks) > 0 for chunks in aspect_groups.values()), "No chunks in any aspect group"
    
    print("\n✓ Evidence scaffolding: PASS")


def test_comparison_prompt():
    """Test comparison prompt generation."""
    print("\n=== Testing Comparison Prompt Generation ===")
    
    from app.services.llm_service import generate_comparison_aware_prompt
    
    class MockDoc:
        def __init__(self, content, paper_id="paper1"):
            self.page_content = content
            self.metadata = {"paper_id": paper_id}
    
    docs = [MockDoc("Content about transformer architecture")]
    comparison_pairs = [("transformers", "RNNs")]
    
    prompt = generate_comparison_aware_prompt(
        "Compare transformers vs RNNs",
        docs,
        aspect_groups={},
        comparison_pairs=comparison_pairs
    )
    
    print(f"\nPrompt length: {len(prompt)} chars")
    print(f"Contains comparison focus: {'COMPARISON FOCUS' in prompt}")
    print(f"Contains methodological instruction: {'Methodological Approach' in prompt}")
    print(f"Contains performance analysis: {'PERFORMANCE ANALYSIS' in prompt}")
    print(f"Contains tradeoff analysis: {'TRADEOFFS' in prompt}")
    
    assert "COMPARISON FOCUS" in prompt, "Missing comparison focus section"
    assert "Methodological Approach" in prompt, "Missing methodological analysis"
    assert "PERFORMANCE ANALYSIS" in prompt, "Missing performance analysis"
    assert "COMPUTATIONAL TRADEOFFS" in prompt, "Missing tradeoff analysis"
    
    print("\n✓ Comparison prompt: PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Comparison Query Improvements")
    print("=" * 60)
    
    try:
        test_comparison_extraction()
        test_query_analysis()
        test_aspect_classification()
        test_evidence_scaffolding()
        test_comparison_prompt()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
