import os
import logging
# Set tokenizers parallelism to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Dict, Any
import json
from dataclasses import dataclass
from enum import Enum
import traceback
import datetime
import time

# Import ModelDefinitions first since other modules depend on it
from LLM import ModelGeneratorConfig, OllamaManager, LLMProvider

from openpyscad import *
from constant import *
from scad_knowledge_base import SCADKnowledgeBase
from OpenSCAD_Generator import OpenSCADGenerator
from KeywordExtractor import KeywordExtractor
from metadata_extractor import MetadataExtractor
from conversation_logger import ConversationLogger
from LLMPromptLogger import LLMPromptLogger

from generator_graph import Model_Generator_Graph
from manual_input_graph import Manual_Knowledge_Graph

# Configure logging
logger = logging.getLogger(__name__)    # Create a logger object using the current module name
logging.basicConfig(filename='debug.log', level=logging.DEBUG)
logger.info("Working on rag_3d_modeler.py")

def clear_logs():
    """Clear both debug log files"""
    open('debug.log', 'w').close()
    open('detailed_debug.log', 'w').close()
    open('detailed_debug.txt', 'w').close()
    logger.info("Logs cleared for new generation")

class GenerationStage(Enum):
    """Enum for different stages of the generation process"""
    INITIALIZATION = "Initialization"
    STEP_BACK = "Step-back Analysis"
    EXAMPLE_SEARCH = "Example Search"
    CODE_GENERATION = "Code Generation"
    VALIDATION = "Validation"
    KNOWLEDGE_UPDATE = "Knowledge Update"

@dataclass
class GenerationSettings:
    """Settings for model generation"""
    similarity_threshold: float = 0.7
    max_generation_attempts: int = 3
    default_resolution: int = 100
    min_code_length: int = 50
    max_code_length: int = 20000


class ErrorHandler:
    """Handler for various types of errors in the generation process"""
    @staticmethod
    def handle_generation_error(error: Exception, attempt: int, max_attempts: int) -> Dict[str, Any]:
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Detailed error logging (to file only)
        logger.error(
            f"\nError Details:"
            f"\n  Type: {error_type}"
            f"\n  Message: {error_msg}"
            f"\n  Attempt: {attempt}/{max_attempts}"
            f"\n  Stack Trace: {traceback.format_exc()}"
        )
        
        # Provide user-friendly error messages
        user_message = {
            "ValueError": "Invalid input parameters",
            "ConnectionError": "Network connection issue",
            "TimeoutError": "Request timed out",
            "RuntimeError": "Execution error",
            "KeyError": "Missing required data",
            "AttributeError": "Invalid operation",
        }.get(error_type, "An unexpected error occurred")
        
        error_info = {
            "success": False,
            "error": f"{user_message}: {error_msg}",
            "attempt": attempt,
            "max_attempts": max_attempts,
            "error_type": error_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "stack_trace": traceback.format_exc()
        }
        
        # Log error details to debug file
        logger.debug(f"Error Info: {json.dumps(error_info, indent=2)}")
        
        # Return simplified error info for terminal display
        return {
            "success": False,
            "error": user_message  # Only show the user-friendly message without technical details
        }

def main():
    """Main function for the 3D Model Generator"""
    config = ModelGeneratorConfig(GenerationSettings())
    # Clear and initialize logging for this session
    clear_logs()
    logger.info("=" * 50)
    logger.info("Starting 3D Model Generator")
    logger.info("Time: %s", datetime.datetime.now().isoformat())
    logger.info("Configuration:")
    logger.info("  - Max Generation Attempts: %d", config.settings.max_generation_attempts)
    logger.info("  - Similarity Threshold: %.2f", config.settings.similarity_threshold)
    logger.info("  - Output Directory: %s", config.output_dir)
    logger.info("=" * 50)
    
    print("Welcome to the 3D Model Generator!")
    
    # Ask for LLM model selection first
    print("\nAvailable LLM Providers:")
    for key, provider in config.llm_providers.items():
        print(f"{key}. {provider['name'].capitalize()} ({provider['model']})")
    
    while True:
        try:
            provider_choice = input("\nSelect LLM provider (1-4, default is 1): ").strip() or "1"
            
            if provider_choice not in config.llm_providers:
                logger.warning(f"Invalid provider choice: {provider_choice}")
                print("Invalid choice. Please select 1, 2, 3, or 4.")
                continue
            
            provider = config.llm_providers[provider_choice]["name"]
            
            # Check Ollama availability for specific providers
            if provider in ["gemma", "deepseek"]:
                logger.info(f"Checking Ollama availability for {provider}")
                if not OllamaManager.check_ollama(provider):
                    continue
            
            logger.info(f"Initializing generator with provider: {provider}")
            break
                
        except Exception as e:
            error_info = ErrorHandler.handle_generation_error(e, 1, 1)
            print(f"Error: {error_info['error']}")
            logger.error("Provider initialization failed: %s", error_info)
            print("Please check your environment variables and try again.")
            return
    
    conversation_logger = ConversationLogger()
    prompt_logger = LLMPromptLogger()
    keyword_extractor = KeywordExtractor()
    metadata_extractor = MetadataExtractor(provider, conversation_logger, prompt_logger)
    kb = SCADKnowledgeBase(keyword_extractor, metadata_extractor, conversation_logger)
    # kb.update_knowledge_base_with_techniques()
    LLM_Provider = LLMProvider()
    
    # Initialize the generator with the selected provider
    try:
        generator = OpenSCADGenerator(provider, knowledge_base=kb, keyword_extractor=keyword_extractor, metadata_extractor=metadata_extractor, conversation_logger=conversation_logger)
        logger.info(f"Successfully initialized generator with provider: {provider}")
    except Exception as e:
        error_info = ErrorHandler.handle_generation_error(e, 1, 1)
        print(f"Error initializing generator: {error_info['error']}")
        logger.error("Generator initialization failed: %s", error_info)
        print("Please check your environment variables and try again.")
        return
    
    # Now continue with the main menu
    while True:
        print("\nSelect an option:")
        print("1. Generate a 3D object")
        print("2. Input knowledge manually")
        print("3. Delete knowledge")
        print("4. View knowledge base")
        print("5. Quit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "5" or choice.lower() in config.quit_words:
            # Log final summary before exiting
            logger.info("User decided to quit the application")
            logger.info("\nSession Summary:")
            logger.info("Session ended at: %s", datetime.datetime.now().isoformat())
            logger.info("=" * 50)
            
            print("\nGoodbye!")
            break
            
        elif choice == "2":
            logger.info("Starting manual knowledge input using new graph")
            
            try:
                # Get LLM instance
                llm = LLM_Provider.get_llm(provider=provider)
                
                # Initialize manual knowledge graph
                manual_graph = Manual_Knowledge_Graph(llm, knowledge_base=kb)
                
                # Handle the complete manual knowledge input process
                result = manual_graph.handle_manual_knowledge_input()
                
                # Log the result
                success = result.get("success", False)
                if success:
                    logger.info("Manual knowledge input successful")
                else:
                    error = result.get("error", "Unknown error")
                    logger.warning(f"Manual knowledge input failed: {error}")
                
            except Exception as e:
                error_info = ErrorHandler.handle_generation_error(e, 1, 1)
                print(f"\nError in manual knowledge input: {error_info['error']}")
                logger.error(f"Manual knowledge input failed: {error_info}")
            
            continue
            
        elif choice == "3":
            logger.info("Starting knowledge deletion")
            result = kb.delete_knowledge()
            logger.info("Knowledge deletion result: %s", "Success" if result else "Failed")
            continue
            
        elif choice == "4":
            logger.info("Starting knowledge base viewer")
            try:
                kb.get_all_examples(interactive=True)
                logger.info("Knowledge base viewing completed")
            except Exception as e:
                error_msg = f"Error viewing knowledge base: {str(e)}"
                logger.error(error_msg)
                print(f"\nError: {error_msg}")
            continue
            
        elif choice == "1":
            
            llm = LLM_Provider.get_llm(provider=provider)
            
            while True:
                user_input = input("Enter a description of the object you want to generate: ")
                if user_input.lower() in config.quit_words:
                    logger.info("User decided to quit the application")
                    break
                else:
                    print(f"User input: {user_input}")  
                
                print("\nProcessing your request. This may take a moment...")
                
                model_generator_graph = Model_Generator_Graph(llm, knowledge_base=kb)
                result = model_generator_graph.generate(user_input)
                
                # Ask if user wants to continue
                continue_input = input("\nWould you like to generate another object? (yes/no): ")
                if continue_input.lower() not in ["yes", "y"]:
                    break
            
        else:
            logger.warning(f"Invalid choice: {choice}")
            print("Invalid choice. Please select 1, 2, 3, 4, or 5.")

if __name__ == "__main__":
    main()