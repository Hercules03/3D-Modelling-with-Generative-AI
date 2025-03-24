from metadata_extractor import MetadataExtractor
from enhanced_scad_knowledge_base import EnhancedSCADKnowledgeBase
import json

def test_metadata_extraction():
    """Test metadata extraction and knowledge base integration"""
    print("\n=== Testing Metadata Extraction ===\n")
    
    # Test description
    description = "Create a modern coffee mug that is 10cm tall with a sleek handle and minimalist design"
    
    print("1. Testing MetadataExtractor directly:\n")
    extractor = MetadataExtractor()
    metadata = extractor.extract_metadata(description)
    
    print("\nExtracted Metadata:")
    print(json.dumps(metadata, indent=2))
    
    print("\n2. Testing Knowledge Base Integration:\n")
    kb = EnhancedSCADKnowledgeBase()
    
    # Add example
    example_added = kb.add_example(description, "")
    print(f"\nExample added successfully: {example_added}")
    
    print("\n3. Retrieving example with metadata:")
    examples = kb.get_relevant_examples(
        "I want a modern minimalist mug",
        similarity_threshold=0.25
    )
    
    print("\nRetrieved Examples:")
    print(json.dumps(examples, indent=2))
    
    kb.cleanup()

if __name__ == "__main__":
    test_metadata_extraction() 