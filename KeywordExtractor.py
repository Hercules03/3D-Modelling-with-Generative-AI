from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from prompts import KEYWORD_EXTRACTOR_SYSTEM_PROMPT, KEYWORD_EXTRACTOR_PROMPT
from llm_management import ModelDefinitions
import json

class KeywordExtractor:
    """Class to extract keywords using LLM"""
    def __init__(self):
        # Initialize LLM lazily - only when needed
        self.llm = None
        self.prompt = KEYWORD_EXTRACTOR_PROMPT
        
        # Basic stop words for minimal filtering
        self.stop_words = {
            'a', 'an', 'the', 'this', 'that', 'create', 'make', 'generate', 
            'model', 'design', 'want', 'need', 'please', 'would', 'like', 'can', 
            'you', 'me', 'build', 'draw', 'sketch', 'i', 'we', 'they', 'he', 'she',
            'of'
        }
        
        self.debug_log = []

    def _init_llm(self):
        """Initialize LLM only when needed"""
        if self.llm is None:
            self.llm = ChatOllama(
                model=ModelDefinitions.KEYWORD_EXTRACTOR,
                temperature=0.0,  # Use 0 temperature for consistent results
                base_url="http://localhost:11434",
                system=KEYWORD_EXTRACTOR_SYSTEM_PROMPT
            )
            self.prompt = KEYWORD_EXTRACTOR_PROMPT

    def _simple_extract(self, description):
        """Simple keyword extraction as fallback"""
        # Convert to lowercase and split
        words = [word for word in description.lower().split() if word not in self.stop_words]
        
        if not words:
            return 'model'
            
        # Return the first non-stop word as core_type
        return words[0]
    def write_debug(self, *messages):
        """Write messages to debug log"""
        for message in messages:
            self.debug_log.append(message)

    def extract_keyword(self, description):
        """Extract the main object keyword and modifiers from the description"""
        try:
            # Initialize LLM for proper keyword extraction
            self._init_llm()
            
            # Replace the placeholder in the prompt
            prompt_value = self.prompt.replace("<<INPUT>>", description)
            self.write_debug(
                "\n=== KEYWORD EXTRACTION PROMPT ===\n",
                f"Description: {description}\n",
                f"Full Prompt Sent to LLM:\n{prompt_value}\n",
                "=" * 50 + "\n"
            )
            print(f"\nSending prompt to extract keywords...")
            
            # Get response from LLM
            response = self.llm.invoke(prompt_value)
            
            # Get the response content and clean it
            if hasattr(response, 'content'):
                content = response.content.strip()
            elif isinstance(response, dict):
                content = response.get('content', response.get('response', '{}'))
            else:
                content = str(response)
                
            # Log the complete LLM response
            self.write_debug(
                "\n=== KEYWORD EXTRACTION RESPONSE ===\n",
                f"Raw Response:\n{content}\n",
                "=" * 50 + "\n"
            )
            
            #print(f"\nRaw response:\n{content}")  # Debug log
            
            # Clean up the content
            content = content.replace('```json', '').replace('```', '').strip()
            
            # print(f"\nCleaned content:\n{content}")  # Debug log
            
            # Try to parse as JSON
            try:
                keyword_data = json.loads(content)
                if not isinstance(keyword_data, dict):
                    raise ValueError("Response is not a JSON object")
                
                result = {
                    "core_type": keyword_data.get("core_type", ""),
                    "modifiers": keyword_data.get("modifiers", []),
                    "compound_type": keyword_data.get("compound_type", "")
                }
                
                # print(f"\nParsed result:\n{json.dumps(result, indent=2)}")  # Debug log
                return result
                
            except json.JSONDecodeError as e:
                print(f"\nJSON parsing failed: {str(e)}")  # Debug log
                # Fall back to simple extraction
                core_type = self._simple_extract(description)
                words = [w for w in description.lower().split() if w not in self.stop_words and w != core_type]
                
                result = {
                    "core_type": core_type,
                    "modifiers": words,
                    "compound_type": f"{' '.join(words)} {core_type}".strip() if words else ""
                }
                
                print(f"\nFallback result:\n{json.dumps(result, indent=2)}")  # Debug log
                return result
            
        except Exception as e:
            print(f"\nExtraction failed: {str(e)}")  # Debug log
            return {
                "core_type": "model",
                "modifiers": [],
                "compound_type": ""
            }