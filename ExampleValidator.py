import os
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
import subprocess
from prompts import VALIDATION_PROMPT
from conversation_logger import ConversationLogger
from LLMmodel import validator_model

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
                model=validator_model,
                temperature=0.1,  # Low temperature for more consistent answers
                base_url="http://localhost:11434"
            )
            self.validation_prompt = ChatPromptTemplate.from_template(VALIDATION_PROMPT)
            self.logger = ConversationLogger()
            self.debug_log = []  # Store debug information
            
        except Exception as e:
            print(f"Error initializing Ollama: {str(e)}")
            print("Please make sure Ollama is installed and running.")
            print("You can install it from: https://ollama.ai")
            print("Then run: ollama pull gemma3:1b")
            raise
    
    def write_debug(self, *messages):
        """Add messages to debug log"""
        self.debug_log.extend(messages)
    
    def save_debug_log(self):
        """Write the complete debug log to debug_val.txt"""
        with open("debug_val.txt", "w") as f:
            f.writelines(self.debug_log)
    
    def extract_object_type(self, description):
        """Extract the main object type from a description"""
        # Convert to lowercase and split into words
        words = description.lower().split()
        
        # Common words to ignore
        ignore_words = {'i', 'want', 'a', 'an', 'to', 'create', 'make', 'build', 'simple', 'basic', 'new'}
        
        # Find the first noun (word not in ignore list)
        for word in words:
            if word not in ignore_words:
                # If it's a compound object (e.g., "coffee cup"), try to get both words
                if len(words) > words.index(word) + 1:
                    next_word = words[words.index(word) + 1]
                    if next_word not in ignore_words:
                        return f"{word} {next_word}"
                return word
        
        return "unknown"
    
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
                # Log the validation decision
                self.logger.log_validation(query, example['description'], "UNUSEFUL")
                # Add to debug log
                self.write_debug("\nChecking example:", comparison, "\n")
                # Print to console
                print(comparison)
                return False
            
            # Format the prompt
            prompt_value = self.validation_prompt.format(
                query=query,
                description=example['description'],
                object_type=object_type
            )
            
            # Add validation attempt to debug log
            self.write_debug(
                "\nChecking example:",
                self.format_comparison(query, example['description'], "Checking..."),
                "\nPrompt:",
                prompt_value,
                "\n"
            )
            
            # Get validation response directly from ChatOllama
            response = self.llm.invoke([{"role": "user", "content": prompt_value}])
            
            # Extract and clean the response
            content = response.content.strip().lower()
            
            # Format the comparison output
            is_useful = content == 'useful'
            decision = "USEFUL" if is_useful else "UNUSEFUL"
            comparison = self.format_comparison(query, example['description'], 
                                             "✓ USEFUL" if is_useful else "✗ UNUSEFUL")
            
            # Log the validation decision
            self.logger.log_validation(query, example['description'], decision)
            
            # Add decision to debug log
            self.write_debug(
                "\nValidation Result:",
                comparison,
                "\n" + "=" * 50 + "\n"
            )
            
            # Debug output to console
            print(comparison)
            
            return is_useful
            
        except Exception as e:
            error_msg = f"Warning: Validation error for example {example.get('filename', 'unknown')}: {str(e)}"
            print(error_msg)
            # Add error to debug log
            self.write_debug(
                "\n=== VALIDATION ERROR ===\n",
                error_msg + "\n",
                "=" * 30 + "\n"
            )
            return False
    
    def filter_relevant_examples(self, query, examples):
        """Filter a list of examples to only include useful ones"""
        if not examples:
            return []
        
        # Clear previous debug log
        self.debug_log = []
        
        # Start new validation session in debug log
        self.write_debug(
            "\n=== STARTING EXAMPLE VALIDATION SESSION ===\n",
            f"Query: {query}\n",
            f"Number of examples to validate: {len(examples)}\n",
            "=" * 50 + "\n"
        )
        
        print(f"\nValidating usefulness of {len(examples)} examples using Gemma3 1B...")
        useful_examples = []
        
        for i, example in enumerate(examples, 1):
            print(f"\nChecking example {i}/{len(examples)}:")
            if self.is_example_relevant(query, example):
                useful_examples.append(example)
        
        # Add validation session summary to debug log
        summary = [
            "\n=== VALIDATION SESSION SUMMARY ===\n",
            f"Total examples checked: {len(examples)}\n",
            f"Examples kept: {len(useful_examples)}\n",
            f"Examples discarded: {len(examples) - len(useful_examples)}\n"
        ]
        
        if useful_examples:
            summary.extend([
                "\nKept Examples:\n",
                *[f"- {ex['description']}\n" for ex in useful_examples]
            ])
        
        summary.append("=" * 50 + "\n\n")
        self.write_debug(*summary)
        
        # Write complete debug log to debug_val.txt
        self.save_debug_log()
        
        if useful_examples:
            print(f"\nKept {len(useful_examples)} useful examples out of {len(examples)} candidates.")
        else:
            print("\nNo useful examples found. Proceeding without examples.")
            
        return useful_examples 