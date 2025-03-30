import os
import traceback
from scad_knowledge_base import SCADKnowledgeBase
from KeywordExtractor import KeywordExtractor
from metadata_extractor import MetadataExtractor
from conversation_logger import ConversationLogger
from LLMPromptLogger import LLMPromptLogger

def main():
    try:
        print("Initializing components...")
        conversation_logger = ConversationLogger()
        prompt_logger = LLMPromptLogger()
        keyword_extractor = KeywordExtractor()
        metadata_extractor = MetadataExtractor("anthropic", conversation_logger, prompt_logger)
        
        print("Initializing knowledge base...")
        kb = SCADKnowledgeBase(
            keyword_extractor=keyword_extractor,
            metadata_extractor=metadata_extractor,
            conversation_logger=conversation_logger
        )
        
        # Test query for propeller
        test_queries = [
            "propeller with 2 blades",
            "a propeller",
            "blade for a fan",
            "model with blades"
        ]
        
        for query in test_queries:
            print("\n" + "="*50)
            print(f"Testing query: '{query}'")
            print("="*50)
            
            # Get raw results with no threshold
            results = kb.get_examples(
                description=query,
                similarity_threshold=0.0,  # No threshold for testing
                max_results=5
            )
            
            print(f"\nFound {len(results)} results")
            
            if results:
                # Print all results
                for i, result in enumerate(results, 1):
                    example = result['example']
                    print(f"\nResult {i}:")
                    print(f"ID: {example.get('id', 'unknown')}")
                    print(f"Score: {result['score']:.3f}")
                    print(f"Object Type: {example['metadata'].get('object_type', 'unknown')}")
                    print(f"Description: {example['metadata'].get('description', 'No description')}")
                    
                    # Print score breakdown
                    print("Score Components:")
                    for component, score in result['score_breakdown']['component_scores'].items():
                        print(f"- {component}: {score:.3f}")
                
                # Modify the KB thresholds
                print("\nApplying recommended thresholds for this system:")
                print("For propeller queries: 0.15")
                print("For blade queries: 0.10")
                
                # Direct fix for the propeller issue
                if 'propeller' in query.lower():
                    relevant_results = [r for r in results if r['score'] >= 0.15]
                elif 'blade' in query.lower():
                    relevant_results = [r for r in results if r['score'] >= 0.10]
                else:
                    relevant_results = [r for r in results if r['score'] >= 0.20]
                
                print(f"\nWith adjusted threshold: {len(relevant_results)} relevant examples")
                
                if relevant_results:
                    top_result = relevant_results[0]
                    example = top_result['example']
                    print(f"\nTop result with adjusted threshold:")
                    print(f"ID: {example.get('id', 'unknown')}")
                    print(f"Score: {top_result['score']:.3f}")
                    print(f"Object Type: {example['metadata'].get('object_type', 'unknown')}")
                    print(f"Description: {example['metadata'].get('description', 'No description')}")
            else:
                print("No examples found for this query")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 