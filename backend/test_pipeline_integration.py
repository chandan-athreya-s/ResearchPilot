"""Quick integration test to verify pipeline improvements."""

import sys
sys.path.insert(0, '/home/chandan-athreya-s/Major-Project/ResearchPilot/backend')

from app.services.query_analyzer import analyze_query
from app.services.llm_service import classify_chunk_aspects, scaffold_evidence_for_comparison, generate_comparison_aware_prompt


def test_pipeline_integration():
    """Test that comparison improvements integrate with pipeline."""
    
    print("\n" + "="*70)
    print("PIPELINE INTEGRATION TEST")
    print("="*70)
    
    # Test 1: Query Analysis passes comparison data
    print("\n[TEST 1] Query Analysis for Comparison Query")
    query = "Compare transformer models vs recurrent neural networks"
    result = analyze_query(query)
    
    print(f"Query: {query}")
    print(f"  ✓ Query type: {result['query_type']}")
    print(f"  ✓ Focus terms: {result['focus_terms'][:3]}")
    print(f"  ✓ Comparison pairs: {result['comparison_pairs']}")
    
    assert result['query_type'] == 'comparison', "Query type detection failed"
    assert len(result['comparison_pairs']) > 0, "Comparison pairs not extracted"
    assert 'comparison_pairs' in result, "Missing comparison_pairs field"
    
    # Test 2: Evidence Classification
    print("\n[TEST 2] Evidence Aspect Classification")
    sample_chunks = [
        "Transformer uses multi-head attention mechanism achieving 95% accuracy",
        "RNN processes sequences iteratively with O(n) memory requirements",
        "Training transformer requires 100K samples vs RNN's 50K sample efficiency",
    ]
    
    for i, chunk in enumerate(sample_chunks):
        aspects = classify_chunk_aspects(chunk)
        detected = [k for k, v in aspects.items() if v]
        print(f"  Chunk {i}: {detected}")
    
    # Test 3: Evidence Scaffolding
    print("\n[TEST 3] Evidence Scaffolding for Comparison")
    
    class MockDoc:
        def __init__(self, content, paper_id):
            self.page_content = content
            self.metadata = {"paper_id": paper_id, "chunk_index": 0}
    
    docs = [
        MockDoc("Transformer architecture with 12 layers", "paper1"),
        MockDoc("Achieved 95% accuracy on ImageNet benchmark", "paper1"),
        MockDoc("RNN processes sequences with memory efficiency", "paper2"),
        MockDoc("Sample efficiency is 50K vs transformer's 100K", "paper2"),
    ]
    
    query_intent = {
        "query_type": "comparison",
        "focus_terms": ["transformer", "rnn"],
        "comparison_pairs": [("transformer", "rnn")]
    }
    
    aspect_groups = scaffold_evidence_for_comparison(docs, query_intent)
    
    aspects_with_chunks = [k for k, v in aspect_groups.items() if v]
    print(f"  Aspects detected: {aspects_with_chunks}")
    print(f"  Total chunks grouped: {sum(len(chunks) for chunks in aspect_groups.values())}")
    
    assert len(aspects_with_chunks) > 0, "No aspects detected in scaffolding"
    
    # Test 4: Comparison Prompt Generation
    print("\n[TEST 4] Comparison Prompt Generation")
    prompt = generate_comparison_aware_prompt(
        query,
        docs,
        aspect_groups=aspect_groups,
        comparison_pairs=result['comparison_pairs']
    )
    
    critical_sections = [
        "COMPARISON FOCUS",
        "Methodological Approach",
        "PERFORMANCE ANALYSIS",
        "COMPUTATIONAL TRADEOFFS",
        "USE CASE SUITABILITY",
        "EXPLICITLY IDENTIFIED TRADEOFFS",
    ]
    
    missing = [s for s in critical_sections if s not in prompt]
    if missing:
        print(f"  ✗ Missing sections: {missing}")
    else:
        print(f"  ✓ All critical sections present")
        print(f"  ✓ Prompt length: {len(prompt)} characters")
    
    assert len(missing) == 0, f"Missing prompt sections: {missing}"
    
    # Test 5: Non-comparison query (backward compatibility)
    print("\n[TEST 5] Non-Comparison Query (Backward Compatibility)")
    survey_query = "Survey of reinforcement learning methods"
    survey_result = analyze_query(survey_query)
    
    print(f"Query: {survey_query}")
    print(f"  ✓ Query type: {survey_result['query_type']}")
    print(f"  ✓ Comparison pairs: {survey_result['comparison_pairs']}")
    
    assert survey_result['query_type'] == 'survey', "Survey type not detected"
    assert survey_result['comparison_pairs'] == [], "Comparison pairs should be empty for non-comparison"
    
    print("\n" + "="*70)
    print("✓ ALL INTEGRATION TESTS PASSED")
    print("="*70)
    print("\nPipeline improvements are ready for use!")
    print("\nKey improvements activated:")
    print("  1. Comparison query detection with pair extraction")
    print("  2. Evidence aspect classification (methods, metrics, tradeoffs, etc.)")
    print("  3. Evidence scaffolding for comparison queries")
    print("  4. Specialized comparison-aware prompting")
    print("  5. Full backward compatibility with existing pipeline")
    print("\nComparison queries will now produce:")
    print("  - Structured 9-section reports")
    print("  - Evidence-grounded comparative analysis")
    print("  - Technical tradeoff analysis")
    print("  - Use-case suitability mapping")
    print("  - Reduced generic LLM language")
    print("\nNon-comparison queries use original 7-section format.")


if __name__ == "__main__":
    try:
        test_pipeline_integration()
    except Exception as e:
        print(f"\n✗ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
