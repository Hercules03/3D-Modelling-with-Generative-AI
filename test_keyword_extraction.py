from metadata_extractor import MetadataExtractor
import json
import os

def test_keyword_extraction():
    print("\n=== Testing Keyword Extraction ===")
    
    # Initialize the metadata extractor
    extractor = MetadataExtractor(llm_provider="gemma")  # Use Gemma as the main provider too
    
    # Test descriptions
    test_descriptions = [
        "I want a pirate sword",
        "Create a modern gaming chair",
        "Make me a decorative vase"
    ]
    
    # Process each test description
    for description in test_descriptions:
        print(f"\nTesting description: {description}")
        metadata = extractor.extract_metadata(description)
        print(f"Returned metadata: {json.dumps(metadata, indent=2)}")
    
    # Verify the log file exists and has content
    log_file = "conversation_logs/keyword_extraction_pairs.json"
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)
            print(f"\nFound {len(logs)} entries in keyword extraction log file")
            for i, log in enumerate(logs, 1):
                print(f"\nEntry {i}:")
                print(json.dumps(log, indent=2))
    else:
        print(f"\nError: Log file {log_file} does not exist!")

if __name__ == "__main__":
    test_keyword_extraction() 