from enhanced_scad_knowledge_base import EnhancedSCADKnowledgeBase

def print_separator():
    print("\n" + "="*80 + "\n")

def test_search_features():
    """Test the enhanced search features"""
    print("\n=== Testing Enhanced Search Features ===\n")
    
    # Initialize knowledge base
    kb = EnhancedSCADKnowledgeBase()
    
    # Test cases with different complexity levels
    test_cases = [
        {
            "query": "I want a simple cylindrical cup",
            "description": "Testing basic shape matching",
            "threshold": 0.2
        },
        {
            "query": "I want a coffee mug with a decorative handle and textured surface",
            "description": "Testing complex design matching",
            "threshold": 0.2
        },
        {
            "query": "I want a modular storage system with interlocking boxes",
            "description": "Testing component-heavy matching",
            "threshold": 0.2
        }
    ]
    
    # Run tests
    for test in test_cases:
        print(f"\nTest: {test['description']}")
        print(f"Query: {test['query']}")
        
        # Get examples with specified threshold
        examples = kb.get_relevant_examples(
            test['query'],
            max_examples=2,
            similarity_threshold=test['threshold']
        )
        
        if examples:
            print("\nFound matching examples:")
            for example in examples:
                print(f"\nDescription: {example['description']}")
                print(f"Scores: {example['scores']}")
                print("\nCode snippet:")
                print("-" * 40)
                print(example['code'][:200] + "..." if len(example['code']) > 200 else example['code'])
                print("-" * 40)
        else:
            print("\nNo matching examples found")
    
    # Cleanup
    kb.cleanup()
    print("\n=== Test Complete ===\n")

if __name__ == "__main__":
    test_search_features() 