import os
from myAPI import *

from langchain_community.vectorstores import Chroma
from openpyscad import *
import subprocess
import json
from constant import *
from SCADKnowledgeBase import KnowledgeBase
from OpenSCAD_Generator import OpenSCADGenerator



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
    file_path = os.path.join(SCAD_KNOWLEDGE_DIR, name)
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
        quit_words = ['quit', 'exit', 'bye', 'q']
        print("\nSelect an option:")
        print("1. Generate a 3D object")
        print("2. Input knowledge manually")
        print("3. Delete knowledge")
        print("4. Quit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "4" or choice.lower() in quit_words:
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
                if description.lower() in quit_words:
                    break
                
                print("I am generating, please be patient...")
                result = generator.generate_model(description)
                if not result['success']:
                    print(f"\nError: {result['error']}")
        
        else:
            print("Invalid choice. Please select 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()