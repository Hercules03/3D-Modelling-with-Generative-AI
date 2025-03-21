from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from prompts import KEYWORD_EXTRACTOR_SYSTEM_PROMPT, KEYWORD_EXTRACTOR_PROMPT
from LLMmodel import keyword_extractor_model

class KeywordExtractor:
    """Class to extract keywords using simple rules first, then falling back to gemma3:1B"""
    def __init__(self):
        # Initialize LLM lazily - only when needed
        self.llm = None
        self.prompt = None
        
        # Common object types and their variations
        self.common_objects = {
            'cup': ['mug', 'glass', 'tumbler', 'goblet'],
            'box': ['container', 'case', 'chest', 'bin'],
            'table': ['desk', 'stand', 'platform'],
            'chair': ['seat', 'stool', 'bench'],
            'bowl': ['dish', 'basin', 'plate'],
            'vase': ['pot', 'vessel', 'urn'],
            'ring': ['band', 'loop', 'circle'],
            'holder': ['stand', 'mount', 'bracket'],
            'frame': ['border', 'edge', 'outline'],
            'base': ['foundation', 'bottom', 'support']
        }
        
        # Words to ignore in descriptions
        self.stop_words = {
            'a', 'an', 'the', 'this', 'that', 'create', 'make', 'generate', 
            'model', 'design', 'want', 'need', 'please', 'would', 'like', 'can', 
            'you', 'me', 'build', 'draw', 'sketch', 'i', 'we', 'they', 'he', 'she',
            'simple', 'basic', 'complex', 'advanced', 'nice', 'good', 'great',
            'beautiful', 'pretty', 'fancy', 'cool', 'awesome', 'amazing'
        }

    def _init_llm(self):
        """Initialize LLM only when needed"""
        if self.llm is None:
            self.llm = ChatOllama(
                model=keyword_extractor_model,
                temperature=0.0,  # Use 0 temperature for consistent results
                base_url="http://localhost:11434",
                system=KEYWORD_EXTRACTOR_SYSTEM_PROMPT
            )
            self.prompt = ChatPromptTemplate.from_template(KEYWORD_EXTRACTOR_PROMPT)

    def _simple_extract(self, description):
        """Simple keyword extraction using rules"""
        # Convert to lowercase and split
        words = description.lower().split()
        
        # Try to find compound objects first (e.g., "coffee cup")
        for i in range(len(words) - 1):
            compound = f"{words[i]} {words[i+1]}"
            # Check if either word is in stop words
            if words[i] not in self.stop_words and words[i+1] not in self.stop_words:
                # Clean the compound
                clean_compound = ''.join(c for c in compound if c.isalnum() or c.isspace())
                if clean_compound:
                    return clean_compound.replace(" ", "_")
        
        # Look for single words that match common objects or their variations
        for word in words:
            if word not in self.stop_words:
                # Check if it's a common object or variation
                for main_obj, variations in self.common_objects.items():
                    if word == main_obj or word in variations:
                        return main_obj
                
                # If not a known object, but not a stop word, use it
                clean_word = ''.join(c for c in word if c.isalnum())
                if clean_word:
                    return clean_word
        
        return 'model'

    def extract_keyword(self, description):
        """Extract the main object keyword from the description"""
        try:
            # Try simple extraction first
            keyword = self._simple_extract(description)
            
            # If we got something other than the default 'model', return it
            if keyword != 'model':
                return keyword
            
            # If simple extraction didn't find a good keyword, try LLM
            self._init_llm()  # Initialize LLM only when needed
            
            # Generate the prompt and get response
            prompt_value = self.prompt.format(description=description)
            response = self.llm.invoke(prompt_value)
            
            # Extract the keyword
            if hasattr(response, 'content'):
                keyword = response.content
            elif isinstance(response, dict):
                keyword = response.get('content', response.get('response', str(response)))
            else:
                keyword = str(response)
            
            # Clean up the keyword
            keyword = keyword.lower().strip()
            keyword = ''.join(c for c in keyword if c.isalnum())
            
            return keyword if keyword else 'model'
            
        except Exception as e:
            print(f"Warning: Keyword extraction failed: {e}")
            return self._simple_extract(description)  # Fall back to simple extraction