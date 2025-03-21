from metadata_extractor import MetadataExtractor
from enhanced_scad_knowledge_base import EnhancedSCADKnowledgeBase
import json

def test_metadata_extraction():
    print("\n=== Testing Metadata Extraction ===\n")
    
    # Test direct metadata extraction
    print("1. Testing MetadataExtractor directly:")
    extractor = MetadataExtractor()
    description = "Create a modern coffee mug that is 10cm tall with a sleek handle and minimalist design"
    metadata = extractor.extract_metadata(description)
    print("\nExtracted Metadata:")
    print(json.dumps(metadata, indent=2))
    
    # Test metadata in knowledge base
    print("\n2. Testing Knowledge Base Integration:")
    kb = EnhancedSCADKnowledgeBase()
    
    # Add an example with metadata
    test_code = """
    difference() {
        cylinder(h=100, r=40);
        translate([0,0,-1])
            cylinder(h=102, r=35);
    }
    """
    
    success = kb.add_example(description, test_code)
    print(f"\nExample added successfully: {success}")
    
    # Retrieve the example with metadata
    print("\n3. Retrieving example with metadata:")
    examples = kb.get_relevant_examples(
        "I want a modern minimalist mug",
        filters={"style": "Modern"}
    )
    print("\nRetrieved Examples:")
    print(examples)
    
    kb.cleanup()

if __name__ == "__main__":
    test_metadata_extraction() 