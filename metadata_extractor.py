from LLM import LLMProvider
import json
from datetime import datetime
from prompts import METADATA_EXTRACTION_PROMPT, KEYWORD_EXTRACTOR_PROMPT, CATEGORY_ANALYSIS_PROMPT
from conversation_logger import ConversationLogger
from LLMmodel import keyword_extractor_model
from LLMPromptLogger import LLMPromptLogger
from constant import BASIC_KNOWLEDGE

class MetadataExtractor:
    def __init__(self, llm_provider="gemma"):
        """Initialize the metadata extractor with an LLM provider"""
        print("\nInitializing Metadata Extractor...")
        # Main LLM for metadata extraction (using Gemma)
        print("- Using gemma3:4b-it-q8_0 for metadata extraction")
        self.llm = LLMProvider.get_llm(
            provider="gemma",
            temperature=0.7,
            model="gemma3:4b-it-q8_0"  # Explicitly use 4b model for main extraction
        )
        print("- Main LLM provider initialized")
        
        self.extraction_prompt = METADATA_EXTRACTION_PROMPT
        self.logger = ConversationLogger()
        self.prompt_logger = LLMPromptLogger()  # Add LLMPromptLogger

    def extract_metadata(self, description, code="", step_back_result=None, keyword_data=None):
        """Extract metadata from a model description"""
        try:
            print(f"\nExtracting metadata")
            
            timestamp = datetime.now().isoformat()
            
            # Handle case where keyword_data is None
            if keyword_data is None:
                print("No keyword data provided, using default values")
                keyword_data = {
                    "core_type": "model",
                    "modifiers": [],
                    "compound_type": ""
                }
            # Also handle string type conversion (in case it's not a dictionary)
            elif isinstance(keyword_data, str):
                print("Converting string keyword_data to dictionary")
                try:
                    keyword_data = json.loads(keyword_data)
                except json.JSONDecodeError:
                    keyword_data = {
                        "core_type": "model",
                        "modifiers": [],
                        "compound_type": ""
                    }
            
            try:
                # Create the analysis object
                analysis = {
                    "query": {
                        "input": description,
                        "timestamp": timestamp,
                        "model": keyword_extractor_model  # Track which model was used
                    },
                    "response": {
                        "core_type": keyword_data.get("core_type", ""),
                        "modifiers": keyword_data.get("modifiers", []),
                        "compound_type": keyword_data.get("compound_type", "")
                    },
                    "metadata": {
                        "success": True,
                        "error": None
                    }
                }
                print(f"Created analysis object: {json.dumps(analysis, indent=2)}")
        
                
                # Use the extracted type for metadata
                object_type = (keyword_data.get("compound_type") or 
                             keyword_data.get("core_type", ""))
                print(f"Using object type: {object_type}")
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse keyword JSON response: {str(e)}")
                print(f"Raw content that failed to parse: {keyword_data}")
                analysis = {
                    "query": {
                        "input": description,
                        "timestamp": timestamp,
                        "model": keyword_extractor_model  # Track which model was used
                    },
                    "response": {
                        "raw_content": keyword_data
                    },
                    "metadata": {
                        "success": False,
                        "error": f"Failed to parse JSON response: {str(e)}"
                    }
                }
                print(f"Created error analysis object: {json.dumps(analysis, indent=2)}")
                self.logger.log_keyword_extraction(analysis)
                object_type = keyword_data.strip()

            
            # Now proceed with full metadata extraction using main LLM
            print("\n=== Starting Full Metadata Extraction ===")
            prompt = self.extraction_prompt.format(description=description, step_back_analysis=step_back_result)
            print("Invoking main LLM for metadata extraction...")
            response = self.llm.invoke(prompt)
            print(f"Raw metadata response: {response.content}")
            
            # Parse JSON response for full metadata
            try:
                # Clean up the response by removing code block markers
                clean_content = response.content.replace("```json", "").replace("```", "").strip()
                metadata = json.loads(clean_content)
                # Update with our enhanced object type
                metadata["object_type"] = object_type
                print("Successfully extracted metadata:")
                for key, value in metadata.items():
                    print(f"  • {key}: {value}")
                
                # Log metadata extraction
                self.prompt_logger.log_metadata_extraction(
                    query=description,
                    code=code,
                    response=metadata,
                    timestamp=timestamp
                )
                
                # Perform category analysis
                category_result = self.analyze_categories(description, metadata)
                if category_result:
                    self.prompt_logger.log_category_analysis(
                        query=description,
                        code=code,
                        response=category_result,
                        timestamp=timestamp
                    )
                
                print("=== Full Metadata Extraction Complete ===\n")
                return metadata
            except json.JSONDecodeError as e:
                print(f"Error parsing metadata response as JSON: {e}")
                print(f"Raw response that failed to parse: {response.content}")
                print("=== Full Metadata Extraction Failed ===\n")
                return {"object_type": object_type}
                
        except Exception as e:
            print(f"Metadata extraction failed with error: {e}")
            print("Stack trace:")
            import traceback
            traceback.print_exc()
            print("=== Metadata Extraction Failed ===\n")
            return {}

    def analyze_categories(self, description, metadata):
        """Analyze categories for the given description and metadata"""
        try:
            # Format the prompt with standard categories and properties
            prompt = CATEGORY_ANALYSIS_PROMPT.format(
                object_type=metadata.get("object_type", ""),
                description=description,
                categories_info=BASIC_KNOWLEDGE["categories"],
                properties_info=BASIC_KNOWLEDGE["properties"]
            )
            
            # Get category analysis from LLM
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            try:
                # Clean up and parse JSON response
                clean_content = content.replace("```json", "").replace("```", "").strip()
                category_data = json.loads(clean_content)
                return category_data
            except json.JSONDecodeError as e:
                print(f"Error parsing category analysis response: {e}")
                return None
                
        except Exception as e:
            print(f"Category analysis failed: {e}")
            return None 