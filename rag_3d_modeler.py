import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.vectorstores import Chroma
from openpyscad import *
import subprocess
import json
import pickle
from pathlib import Path
import datetime
from prompts import BASIC_KNOWLEDGE, OLLAMA_SYSTEM_PROMPT, KEYWORD_EXTRACTOR_PROMPT, KEYWORD_EXTRACTOR_SYSTEM_PROMPT, OPENSCAD_GNERATOR_PROMPT_TEMPLATE

# Load environment variables
load_dotenv()

# Constants
KNOWLEDGE_DIR = "knowledge_base"
CHROMA_PERSIST_DIR = os.path.join(KNOWLEDGE_DIR, "chroma")
EXAMPLES_FILE = os.path.join(KNOWLEDGE_DIR, "examples.pkl")



class LLMProvider:
    """Class to manage different LLM providers"""
    
    @staticmethod
    def get_llm(provider="anthropic", temperature=0.7):
        """
        Get LLM instance based on provider
        Args:
            provider (str): 'anthropic', 'openai', 'gemma', 'deepseek'
            temperature (float): Temperature for generation
        Returns:
            LLM instance
        """
        provider = provider.lower()
        
        # Get API key and base URL from environment
        api_key = os.getenv("API_KEY")
        base_url = os.getenv("BASE_URL")
        
        if provider in ["anthropic", "openai"]:
            if not api_key or not base_url:
                raise ValueError("API_KEY and BASE_URL must be set in environment variables")
        
        if provider == "anthropic":
            base_url = "https://api2.qyfxw.cn/v1"  # Fixed base URL
            return ChatOpenAI(
                # model = "claude-3-5-sonnet-20240620",
                model="claude-3-7-sonnet-20250219",
                temperature=temperature,
                openai_api_key=api_key,
                base_url=base_url
            )
            
        elif provider == "openai":
            # Clean up base URL to prevent path duplication
            base_url = "https://api2.qyfxw.cn/v1"  # Fixed base URL
            return ChatOpenAI(
                model="o1-mini",
                temperature=1.0,  # O1-Mini only supports temperature=1.0
                openai_api_key=api_key,
                base_url=base_url
            )
            
        elif provider == "gemma":
            return ChatOllama(
                model="gemma3:4b-it-q8_0",
                temperature=temperature,
                base_url="http://localhost:11434",
                stop=None,
                streaming=False,
                seed=None,
                system=OLLAMA_SYSTEM_PROMPT
            )
        elif provider == "deepseek":
            return ChatOllama(
                model="deepseek-r1:7b",
                temperature=temperature,
                base_url="http://localhost:11434",
                stop=None,
                streaming=False,
                seed=None,
                system=OLLAMA_SYSTEM_PROMPT
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

class KeywordExtractor:
    """Class to extract keywords using Llama 3.2"""
    def __init__(self):
        self.llm = ChatOllama(
            model="llama3.2:1b",
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

class KnowledgeBase:
    def __init__(self):
        """Initialize the knowledge base"""
        self.examples = []
        self.keyword_extractor = KeywordExtractor()
        self.load_examples()
        
    def load_examples(self):
        """Load existing examples from files"""
        self.examples = []
        if os.path.exists(KNOWLEDGE_DIR):
            for file in os.listdir(KNOWLEDGE_DIR):
                if file.endswith('.pkl'):
                    try:
                        with open(os.path.join(KNOWLEDGE_DIR, file), 'rb') as f:
                            example = pickle.load(f)
                            self.examples.append(example)
                    except Exception as e:
                        print(f"Error loading example {file}: {e}")
    
    def get_next_filename(self, base_name):
        """Get the next available numbered filename for a given base name"""
        if not os.path.exists(KNOWLEDGE_DIR):
            return f"{base_name}1.pkl"
            
        # Find all existing files with the same base name
        existing_files = [f for f in os.listdir(KNOWLEDGE_DIR) 
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
        os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
        
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
            filepath = os.path.join(KNOWLEDGE_DIR, filename)
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
        if not self.examples:
            return ""
        
        # Get base name from query
        query_base = self.get_base_name(query)
        
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
        
        examples_text = "\nRelevant examples:\n"
        for ex in relevant:
            examples_text += f"\nDescription: {ex['description']}\nCode:\n{ex['code']}\n"
        return examples_text

class OpenSCADGenerator:
    def __init__(self, llm_provider="anthropic"):
        """
        Initialize the OpenSCAD generator
        Args:
            llm_provider (str): LLM provider to use ('anthropic', 'openai', or 'gemma')
        """
        self.llm = LLMProvider.get_llm(llm_provider)
        self.llm_provider = llm_provider
        self.knowledge_base = KnowledgeBase()
        
        # Create prompt template
        template = OPENSCAD_GNERATOR_PROMPT_TEMPLATE
        
        self.prompt = ChatPromptTemplate.from_template(template)
        
    def generate_model(self, description):
        """Generate OpenSCAD code for the given description and save it to a file."""
        try:
            # Get relevant examples from knowledge base
            examples = self.knowledge_base.get_relevant_examples(description)
            
            # Prepare the prompt inputs
            inputs = {
                "basic_knowledge": BASIC_KNOWLEDGE,
                "examples": examples,
                "request": description
            }
            
            # Generate the prompt and get response
            prompt_value = self.prompt.format(**inputs)
            response = self.llm.invoke(prompt_value)
            
            # Save raw response to debug file
            with open("debug.txt", "w") as f:
                f.write(f"Provider: {self.llm_provider}\n")
                f.write(f"Prompt:\n{prompt_value}\n\n")
                f.write(f"Raw Response:\n{str(response)}\n")
            
            # Initialize reasoning variable
            reasoning_process = None
            
            # Extract code from response based on provider
            if self.llm_provider == "gemma":
                # Handle gemma's response format
                if hasattr(response, 'content'):
                    scad_code = response.content
                elif isinstance(response, dict):
                    scad_code = response.get('content', response.get('response', str(response)))
                else:
                    scad_code = str(response)
            elif self.llm_provider == "openai":
                # Handle OpenAI's response format
                if hasattr(response, 'response_metadata'):
                    metadata = response.response_metadata
                    if 'token_usage' in metadata and 'completion_tokens_details' in metadata['token_usage']:
                        reasoning_process = metadata['token_usage']['completion_tokens_details'].get('reasoning_tokens')
                scad_code = str(response)
            elif self.llm_provider == "anthropic":
                # Handle Anthropic's response format
                if hasattr(response, 'content'):
                    scad_code = response.content
                else:
                    scad_code = str(response)
                
                # Extract the actual code content
                if hasattr(response, 'content'):
                    content = response.content
                    # Remove markdown code block if present
                    if content.startswith('```') and content.endswith('```'):
                        lines = content.split('\n')
                        if lines[0].startswith('```'):
                            lines = lines[1:]
                        if lines[-1].startswith('```'):
                            lines = lines[:-1]
                        scad_code = '\n'.join(lines)
                    else:
                        scad_code = content
                else:
                    scad_code = str(response)
            elif self.llm_provider == "deepseek":
                # Handle DeepSeek's response format
                if hasattr(response, 'content'):
                    content = response.content
                else:
                    content = str(response)
                
                # Extract thinking/reasoning process if present
                if '<think>' in content and '</think>' in content:
                    think_start = content.find('<think>') + len('<think>')
                    think_end = content.find('</think>')
                    reasoning_process = content[think_start:think_end].strip()
                    # Remove the thinking section from the code
                    content = content[:think_start-len('<think>')] + content[think_end+len('</think>'):]
                
                # Clean up the content
                if 'content=' in content:
                    # Extract content between content=" and the next quotation mark
                    content_start = content.find('content="') + len('content="')
                    content_end = content.find('"', content_start)
                    if content_end != -1:
                        content = content[content_start:content_end]
                
                # Unescape newlines
                content = content.replace('\\n', '\n')
                
                # Look for code block markers
                if '```' in content:
                    # Find the first and last code block
                    blocks = content.split('```')
                    for block in blocks:
                        # Skip empty blocks or blocks that start with plaintext/scad
                        if block.strip() and not block.strip().startswith('plaintext') and not block.strip().startswith('scad'):
                            scad_code = block.strip()
                            break
                    else:
                        # If no suitable block found, use the cleaned content
                        scad_code = content.strip()
                else:
                    # If no code block markers, clean up the content
                    scad_code = content.strip()
                
                # Update debug file with parsed components
                with open("debug.txt", "a") as f:
                    f.write("\nParsed Components:\n")
                    f.write(f"Reasoning Process:\n{reasoning_process}\n\n")
                    f.write(f"Extracted SCAD Code:\n{scad_code}\n")
            else:
                scad_code = str(response)
            
            # Clean up any empty responses
            if not scad_code or scad_code.strip() == '':
                raise ValueError("Generated code is empty or invalid")
            
            # Save to file
            with open("output.scad", "w") as f:
                f.write(scad_code)
            
            print("\nOpenSCAD code has been generated and saved to 'output.scad'")
            print("\nGenerated Code:")
            print("-" * 40)
            print(scad_code)
            print("-" * 40)
            
            """
            # If there was reasoning, display it
            if reasoning_process:
                print("\nModel's Reasoning Process:")
                print("-" * 40)
                print(reasoning_process)
                print("-" * 40)
            """
            
            # Ask user if they want to add this to the knowledge base
            add_to_kb = input("\nWould you like to add this example to the knowledge base? (y/n): ").lower().strip()
            if add_to_kb == 'y':
                self.knowledge_base.add_example(description, scad_code)
                print("Example added to knowledge base!")
            
            return True
        except Exception as e:
            print(f"Error generating model: {str(e)}")
            print("\nDebug - Exception details:", e)
            # Write error to debug file
            with open("debug.txt", "a") as f:
                f.write(f"\nError occurred:\n{str(e)}\n")
            return False

def check_ollama(llm_provider):
    """Check if Ollama is installed and running"""
    try:
        response = subprocess.run(['curl', '-s', 'http://localhost:11434/api/tags'], capture_output=True, text=True)
        if response.returncode == 0:
            # Check if the required model is available
            if llm_provider == "gemma":
                model_name = "gemma3:4b-it-q8_0"
            elif llm_provider == "deepseek":
                model_name = "deepseek-r1:7b"
            else:
                model_name = "gemma3:4b-it-q8_0"
            model_list = json.loads(response.stdout)
            if not any(model['name'] == model_name for model in model_list['models']):
                print(f"\nWarning: {model_name} not found. Please run: ollama pull {model_name}")
                return False
            return True
        return False
    except Exception:
        return False

def input_manual_knowledge():
    """Function to handle manual input of knowledge"""
    print("\nManual Knowledge Input Mode")
    print("---------------------------")
    
    # Get the description/query
    description = input("\nEnter the description/query for the 3D model: ").strip()
    if not description:
        print("Description cannot be empty.")
        return False
        
    # Get the OpenSCAD code
    print("\nEnter the OpenSCAD code (press Enter twice to finish):")
    print("----------------------------------------------------")
    
    lines = []
    while True:
        line = input()
        if line.strip() == "" and (not lines or lines[-1].strip() == ""):
            break
        lines.append(line)
    
    scad_code = "\n".join(lines[:-1])  # Remove the last empty line
    
    if not scad_code.strip():
        print("OpenSCAD code cannot be empty.")
        return False
    
    # Initialize knowledge base and save the example
    try:
        kb = KnowledgeBase()
        kb.add_example(description, scad_code)
        print("\nKnowledge has been successfully saved!")
        return True
    except Exception as e:
        print(f"\nError saving knowledge: {str(e)}")
        return False

def delete_knowledge():
    """Function to handle deletion of knowledge"""
    print("\nDelete Knowledge Mode")
    print("--------------------")
    
    # Get the knowledge name
    name = input("\nEnter the name of the knowledge to delete (e.g., 'snowman1' for snowman1.pkl): ").strip()
    if not name:
        print("Name cannot be empty.")
        return False
    
    # Add .pkl extension if not provided
    if not name.endswith('.pkl'):
        name = f"{name}.pkl"
    
    # Check if file exists
    file_path = os.path.join(KNOWLEDGE_DIR, name)
    if not os.path.exists(file_path):
        print(f"\nError: Knowledge file '{name}' not found.")
        return False
    
    # Confirm deletion
    confirm = input(f"\nAre you sure you want to delete '{name}'? (y/n): ").lower().strip()
    if confirm != 'y':
        print("\nDeletion cancelled.")
        return False
    
    # Delete the file
    try:
        os.remove(file_path)
        print(f"\nKnowledge file '{name}' has been successfully deleted!")
        return True
    except Exception as e:
        print(f"\nError deleting knowledge file: {str(e)}")
        return False

def main():
    print("Welcome to the 3D Model Generator!")
    
    while True:
        print("\nSelect an option:")
        print("1. Generate a 3D object")
        print("2. Input knowledge manually")
        print("3. Delete knowledge")
        print("4. Quit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "4":
            print("\nGoodbye!")
            break
            
        elif choice == "2":
            input_manual_knowledge()
            continue
            
        elif choice == "3":
            delete_knowledge()
            continue
            
        elif choice == "1":
            print("\nAvailable LLM Providers:")
            print("1. Anthropic (Claude-3-Sonnet)")
            print("2. OpenAI (O1-Mini)")
            print("3. Gemma3:4B")
            print("4. DeepSeek-R1:7B")
            
            while True:
                try:
                    provider_choice = input("\nSelect LLM provider (1-4, default is 1): ").strip()
                    
                    if provider_choice == "":
                        provider_choice = "1"
                        
                    if provider_choice not in ["1", "2", "3", "4"]:
                        print("Invalid choice. Please select 1, 2, 3, or 4.")
                        continue
                    
                    provider = {
                        "1": "anthropic",
                        "2": "openai",
                        "3": "gemma",
                        "4": "deepseek"
                    }[provider_choice]
                    
                    # Check if Ollama is available when selected
                    if provider == "gemma" and not check_ollama("gemma"):
                        print("Ollama is not installed or not running. Please install and start Ollama first.")
                        print("Make sure to run: ollama pull gemma3:4b-it-q8_0")
                        continue
                    elif provider == "deepseek" and not check_ollama("deepseek"):
                        print("Ollama is not installed or not running. Please install and start Ollama first.")
                        print("Make sure to run: ollama pull deepseek-r1:7b")
                        continue
                    
                    # Initialize generator with selected provider
                    generator = OpenSCADGenerator(provider)
                    break
                    
                except Exception as e:
                    print(f"Error: {str(e)}")
                    print("Please check your environment variables and try again.")
                    return
            
            print("\nDescribe the 3D object you want to create, and I'll generate OpenSCAD code for it.")
            print("Type 'quit' to exit.")
            
            while True:
                description = input("\nWhat would you like to model? ")
                quit_words = ['quit', 'exit', 'bye', 'q']
                if description.lower() in quit_words:
                    break
                
                print("I am generating, please be patient...")
                generator.generate_model(description)
        
        else:
            print("Invalid choice. Please select 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()