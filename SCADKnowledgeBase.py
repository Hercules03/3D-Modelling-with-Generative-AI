from constant import *
from KeywordExtractor import KeywordExtractor
from ExampleValidator import ExampleValidator
import os
import pickle
import datetime

class KnowledgeBase:
    def __init__(self):
        """Initialize the knowledge base"""
        self.examples = []
        self.keyword_extractor = KeywordExtractor()
        self.example_validator = ExampleValidator()
        self.load_examples()
        
    def load_examples(self):
        """Load existing examples from files"""
        self.examples = []
        if os.path.exists(SCAD_KNOWLEDGE_DIR):
            for file in os.listdir(SCAD_KNOWLEDGE_DIR):
                if file.endswith('.pkl'):
                    try:
                        with open(os.path.join(SCAD_KNOWLEDGE_DIR, file), 'rb') as f:
                            example = pickle.load(f)
                            self.examples.append(example)
                    except Exception as e:
                        print(f"Error loading example {file}: {e}")
    
    def get_next_filename(self, base_name):
        """Get the next available numbered filename for a given base name"""
        if not os.path.exists(SCAD_KNOWLEDGE_DIR):
            return f"{base_name}1.pkl"
            
        # Find all existing files with the same base name
        existing_files = [f for f in os.listdir(SCAD_KNOWLEDGE_DIR) 
                         if f.startswith(base_name) and f.endswith('.pkl')]
        
        if not existing_files:
            return f"{base_name}1.pkl"
            
        # Extract numbers from existing filenames
        numbers = []
        for filename in existing_files:
            try:
                # Extract number between base_name and .pkl
                num = int(filename[len(base_name):-4])
                numbers.append(num)
            except ValueError:
                continue
        
        # If no valid numbers found, start with 1
        if not numbers:
            return f"{base_name}1.pkl"
            
        # Return next number after the maximum
        return f"{base_name}{max(numbers) + 1}.pkl"
    
    def get_base_name(self, description):
        """Extract a base name from the description using Llama"""
        return self.keyword_extractor.extract_keyword(description)
    
    def save_example(self, description, code):
        """Save example to a numbered file based on object type"""
        os.makedirs(SCAD_KNOWLEDGE_DIR, exist_ok=True)
        
        # Get base name from description
        base_name = self.get_base_name(description)
        
        # Get next available filename
        filename = self.get_next_filename(base_name)
        
        # Create example data
        example = {
            'description': description,
            'code': code,
            'filename': filename,
            'created_at': datetime.datetime.now().isoformat()
        }
        
        try:
            # Save to file
            filepath = os.path.join(SCAD_KNOWLEDGE_DIR, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(example, f)
            
            # Add to in-memory list
            self.examples.append(example)
            print(f"\nExample saved as: {filename}")
            
        except Exception as e:
            print(f"Error saving example: {e}")
    
    def add_example(self, description, code):
        """Add a new example to the knowledge base"""
        self.save_example(description, code)
    
    def get_relevant_examples(self, query, n=3):
        """Get relevant examples based on description similarity"""
        print("\nRetrieving knowledge from SCAD knowledge base...")
        
        if not self.examples:
            print("No examples found in SCAD knowledge base.")
            return ""
        
        # Extract main object type using KeywordExtractor
        query_object = self.keyword_extractor.extract_keyword(query)
        print(f"Searching for examples related to: {query_object}")
        
        # Score and rank all examples based on relevance
        scored_examples = []
        for ex in self.examples:
            score = 0
            desc_words = set(ex['description'].lower().split())
            query_words = set(query.lower().split())
            
            # Check for exact object match
            if query_object in desc_words:
                score += 3
            
            # Check for compound matches (e.g., "coffee cup" for "cup")
            if any(query_object in ' '.join(pair) 
                  for pair in zip(desc_words, list(desc_words)[1:])):
                score += 2
            
            # Check for word overlap (excluding common words)
            common_words = {'i', 'want', 'a', 'an', 'to', 'create', 'make', 'build', 
                          'simple', 'basic', 'new', 'the', 'that', 'this', 'it'}
            meaningful_query_words = query_words - common_words
            meaningful_desc_words = desc_words - common_words
            word_overlap = len(meaningful_query_words & meaningful_desc_words)
            score += word_overlap
            
            if score > 0:  # Only include examples with some relevance
                scored_examples.append((score, ex))
        
        # Sort by score (highest first) and take top n
        scored_examples.sort(reverse=True, key=lambda x: x[0])
        candidates = [ex for score, ex in scored_examples[:n] if score > 0]
        
        if not candidates:
            print("No relevant examples found.")
            return ""
            
        # Validate usefulness using Gemma3
        useful = self.example_validator.filter_relevant_examples(query, candidates)
        
        if not useful:
            print("No examples passed usefulness validation.")
            return ""
            
        examples_text = "\nRelevant examples:\n"
        for ex in useful:
            examples_text += f"\nDescription: {ex['description']}\nCode:\n{ex['code']}\n"
        
        return examples_text