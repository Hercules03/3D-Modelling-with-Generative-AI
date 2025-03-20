from constant import *
from KeywordExtractor import KeywordExtractor
import os
import pickle
import datetime

class KnowledgeBase:
    def __init__(self):
        """Initialize the knowledge base"""
        self.examples = []
        self.keyword_extractor = KeywordExtractor()
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
        
        # Get base name from query
        query_base = self.get_base_name(query)
        print(f"Searching for examples related to: {query_base}")
        
        # First, try to find examples with the same base name
        relevant = [ex for ex in self.examples 
                   if ex['filename'].startswith(query_base)]
        
        # If we don't have enough, add other recent examples
        if len(relevant) < n:
            other_examples = [ex for ex in self.examples 
                            if ex not in relevant][-n + len(relevant):]
            relevant.extend(other_examples)
        
        # Take the most recent n examples
        relevant = relevant[-n:]
        
        if relevant:
            print(f"{len(relevant)} relevant examples found.")
        else:
            print("No relevant examples found.")
        
        examples_text = "\nRelevant examples:\n"
        for ex in relevant:
            examples_text += f"\nDescription: {ex['description']}\nCode:\n{ex['code']}\n"
        return examples_text