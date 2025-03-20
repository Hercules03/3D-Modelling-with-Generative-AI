from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
import subprocess
from prompts import VALIDATION_PROMPT


class ExampleValidator:
    def __init__(self):
        """Initialize the validator with Gemma3 1B model"""
        # Check if Ollama is running
        try:
            response = subprocess.run(['curl', '-s', 'http://localhost:11434/api/tags'], capture_output=True, text=True)
            if response.returncode != 0:
                raise Exception("Ollama service is not running")
                
            # Initialize ChatOllama with specific model
            self.llm = ChatOllama(
                model="gemma3:1b",
                temperature=0.1,  # Low temperature for more consistent answers
                base_url="http://localhost:11434"
            )
            self.validation_prompt = ChatPromptTemplate.from_template(VALIDATION_PROMPT)
            
        except Exception as e:
            print(f"Error initializing Ollama: {str(e)}")
            print("Please make sure Ollama is installed and running.")
            print("You can install it from: https://ollama.ai")
            print("Then run: ollama pull gemma3:1b")
            raise
    
    def extract_object_type(self, description):
        """Extract the main object type from a description"""
        # Simple extraction of the first noun or object mentioned
        words = description.lower().split()
        return next((word for word in words if word.isalnum()), "unknown")
    
    def format_comparison(self, query, example_desc, decision):
        """Format the comparison between query and example"""
        separator = "-" * 50
        return f"""
{separator}
Desired Object:  {query}
Example Object:  {example_desc}
Decision:        {decision}
{separator}"""
    
    def is_example_relevant(self, query, example):
        """Check if an example would be useful for the query"""
        try:
            # Extract object type from description
            object_type = self.extract_object_type(example['description'])
            
            # Basic keyword matching first
            query_words = set(query.lower().split())
            desc_words = set(example['description'].lower().split())
            if not any(word in desc_words for word in query_words if len(word) > 2):
                comparison = self.format_comparison(query, example['description'], "✗ UNUSEFUL (No keyword match)")
                # Write to debug file first
                with open("debug.txt", "a") as f:
                    f.write("\nChecking example:\n")
                    f.write(comparison)
                    f.write("\n")
                # Then print to console
                print(comparison)
                return False
            
            # Format the prompt
            prompt_value = self.validation_prompt.format(
                query=query,
                description=example['description'],
                object_type=object_type
            )
            
            # Write validation attempt to debug file
            with open("debug.txt", "a") as f:
                f.write("\nChecking example:\n")
                f.write(self.format_comparison(query, example['description'], "Checking..."))
                f.write("\nPrompt:\n")
                f.write(prompt_value)
                f.write("\n")
            
            # Get validation response directly from ChatOllama
            response = self.llm.invoke([{"role": "user", "content": prompt_value}])
            
            # Extract and clean the response
            content = response.content.strip().lower()
            
            # Format the comparison output
            comparison = self.format_comparison(query, example['description'], 
                                             "✓ USEFUL" if content == "useful" else "✗ UNUSEFUL")
            
            # Write decision to debug file
            with open("debug.txt", "a") as f:
                f.write("\nValidation Result:\n")
                f.write(comparison)
                f.write("\n" + "=" * 50 + "\n")
            
            # Debug output to console
            print(comparison)
            
            return content == 'useful'
            
        except Exception as e:
            error_msg = f"Warning: Validation error for example {example.get('filename', 'unknown')}: {str(e)}"
            print(error_msg)
            # Write error to debug file
            with open("debug.txt", "a") as f:
                f.write("\n=== VALIDATION ERROR ===\n")
                f.write(error_msg + "\n")
                f.write("=" * 30 + "\n")
            return False
    
    def filter_relevant_examples(self, query, examples):
        """Filter a list of examples to only include useful ones"""
        if not examples:
            return []
        
        # Write validation session start to debug file
        with open("debug.txt", "a") as f:
            f.write("\n=== STARTING EXAMPLE VALIDATION SESSION ===\n")
            f.write(f"Query: {query}\n")
            f.write(f"Number of examples to validate: {len(examples)}\n")
            f.write("=" * 50 + "\n")
            
        print(f"\nValidating usefulness of {len(examples)} examples using Gemma3 1B...")
        useful_examples = []
        
        for i, example in enumerate(examples, 1):
            print(f"\nChecking example {i}/{len(examples)}:")
            if self.is_example_relevant(query, example):
                useful_examples.append(example)
        
        # Write validation session summary to debug file
        with open("debug.txt", "a") as f:
            f.write("\n=== VALIDATION SESSION SUMMARY ===\n")
            f.write(f"Total examples checked: {len(examples)}\n")
            f.write(f"Examples kept: {len(useful_examples)}\n")
            f.write(f"Examples discarded: {len(examples) - len(useful_examples)}\n")
            if useful_examples:
                f.write("\nKept Examples:\n")
                for ex in useful_examples:
                    f.write(f"- {ex['description']}\n")
            f.write("=" * 50 + "\n\n")
        
        if useful_examples:
            print(f"\nKept {len(useful_examples)} useful examples out of {len(examples)} candidates.")
        else:
            print("\nNo useful examples found. Proceeding without examples.")
            
        return useful_examples 