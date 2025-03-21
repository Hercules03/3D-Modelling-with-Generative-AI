from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from prompts import KEYWORD_EXTRACTOR_SYSTEM_PROMPT, KEYWORD_EXTRACTOR_PROMPT
from LLMmodel import keyword_extractor_model
class KeywordExtractor:
    """Class to extract keywords using Llama 3.2"""
    def __init__(self):
        self.llm = ChatOllama(
            model=keyword_extractor_model,
            temperature=0.0,  # Use 0 temperature for consistent results
            base_url="http://localhost:11434",
            system=KEYWORD_EXTRACTOR_SYSTEM_PROMPT
        )
        
        self.prompt = ChatPromptTemplate.from_template(KEYWORD_EXTRACTOR_PROMPT)

    def extract_keyword(self, description):
        """Extract the main object keyword from the description"""
        try:
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
            # Fall back to simple extraction if Llama fails
            return self._simple_extract(description)
    
    def _simple_extract(self, description):
        """Fallback method for simple keyword extraction"""
        words = description.lower().split()
        stop_words = {
            'a', 'an', 'the', 'this', 'that', 'create', 'make', 'generate', 
            'model', 'design', 'want', 'need', 'please', 'would', 'like', 'can', 
            'you', 'me', 'build', 'draw', 'sketch', 'i', 'we', 'they', 'he', 'she'
        }
        for word in words:
            if word not in stop_words:
                base_name = ''.join(c for c in word if c.isalnum())
                if base_name:
                    return base_name
        return 'model'