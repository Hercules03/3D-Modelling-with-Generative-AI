from enhanced_scad_knowledge_base import EnhancedSCADKnowledgeBase

def test_similarity_threshold():
    print("\n=== Testing Similarity Threshold ===\n")
    
    kb = EnhancedSCADKnowledgeBase()
    
    # Test queries with different thresholds
    test_cases = [
        {
            "query": "I want a modern coffee mug",
            "threshold": 0.4,
            "description": "Low threshold (0.4)"
        },
        {
            "query": "I want a modern coffee mug",
            "threshold": 0.3,
            "description": "Very low threshold (0.3)"
        },
        {
            "query": "I want a coffee cup",  # Different phrasing
            "threshold": 0.4,
            "description": "Different query phrasing"
        }
    ]
    
    for test in test_cases:
        print(f"\nTesting with {test['description']}:")
        print(f"Query: {test['query']}")
        print(f"Threshold: {test['threshold']}")
        print("-" * 50)
        
        examples = kb.get_relevant_examples(
            test['query'],
            similarity_threshold=test['threshold']
        )
        
        print("\nResults:")
        print(examples if examples else "No examples found")
        print("=" * 50)
    
    kb.cleanup()

if __name__ == "__main__":
    test_similarity_threshold() 