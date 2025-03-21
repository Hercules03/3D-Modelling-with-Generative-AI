from LLM import LLMProvider
import json
from prompts import METADATA_EXTRACTION_PROMPT
class MetadataExtractor:
    def __init__(self, llm_provider="anthropic"):
        """Initialize the metadata extractor with an LLM provider"""
        print("\nInitializing Metadata Extractor...")
        self.llm = LLMProvider.get_llm(provider=llm_provider)
        print("- LLM provider initialized")
        
        self.extraction_prompt = METADATA_EXTRACTION_PROMPT

    def extract_metadata(self, description):
        """Extract metadata from a model description"""
        try:
            print(f"\nExtracting metadata from description...")
            
            # Format and send prompt to LLM
            prompt = self.extraction_prompt.format(description=description)
            response = self.llm.invoke(prompt)
            
            # Parse JSON response
            try:
                metadata = json.loads(response.content)
                print("- Successfully extracted metadata:")
                for key, value in metadata.items():
                    print(f"  • {key}: {value}")
                return metadata
            except json.JSONDecodeError as e:
                print(f"- Error parsing LLM response as JSON: {e}")
                print("- Raw response:", response.content)
                return {}
                
        except Exception as e:
            print(f"- Metadata extraction failed: {e}")
            return {} 