import os
import logging
from typing import Dict, Optional, List, Any
import subprocess
import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from myAPI import *
from langchain_community.vectorstores import Chroma
from openpyscad import *
from constant import *
from enhanced_scad_knowledge_base import EnhancedSCADKnowledgeBase
from OpenSCAD_Generator import OpenSCADGenerator
from LLMmodel import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    max_code_length: int = 10000

class ModelGeneratorConfig:
    """Configuration class for the 3D Model Generator"""
    def __init__(self):
        self.quit_words = ['quit', 'exit', 'bye', 'q']
        self.llm_providers = {
            "1": {"name": "anthropic", "model": anthropic_model},
            "2": {"name": "openai", "model": openai_model},
            "3": {"name": "gemma", "model": gemma_model},
            "4": {"name": "deepseek", "model": deepseek_model}
        }
        self.settings = GenerationSettings()
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

class ErrorHandler:
    """Handler for various types of errors in the generation process"""
    @staticmethod
    def handle_generation_error(error: Exception, attempt: int, max_attempts: int) -> Dict[str, Any]:
        error_type = type(error).__name__
        error_msg = str(error)
        
        logger.error(f"Generation attempt {attempt}/{max_attempts} failed: [{error_type}] {error_msg}")
        
        # Provide user-friendly error messages
        user_message = {
            "ValueError": "Invalid input parameters",
            "ConnectionError": "Network connection issue",
            "TimeoutError": "Request timed out",
        }.get(error_type, "An unexpected error occurred")
        
        return {
            "success": False,
            "error": f"{user_message}: {error_msg}",
            "attempt": attempt,
            "max_attempts": max_attempts,
            "error_type": error_type
        }

class ProgressManager:
    """Manager for tracking and reporting generation progress"""
    def __init__(self):
        self._current_stage: Optional[GenerationStage] = None
        self._progress: float = 0.0
        self._stage_weights = {
            GenerationStage.INITIALIZATION: 0.1,
            GenerationStage.STEP_BACK: 0.2,
            GenerationStage.EXAMPLE_SEARCH: 0.2,
            GenerationStage.CODE_GENERATION: 0.3,
            GenerationStage.VALIDATION: 0.1,
            GenerationStage.KNOWLEDGE_UPDATE: 0.1
        }

    def update_stage(self, stage: GenerationStage, progress: float, message: str):
        """Update progress for the current stage"""
        self._current_stage = stage
        self._progress = progress
        total_progress = self._calculate_total_progress()
        logger.info(f"[{stage.value}] {progress:.1f}% - {message}")
        print(f"\rProgress: [{self._create_progress_bar(total_progress)}] {total_progress:.1f}%", end="")

    def _calculate_total_progress(self) -> float:
        """Calculate total progress across all stages"""
        if not self._current_stage:
            return 0.0
        
        completed_weight = sum(
            weight for stage, weight in self._stage_weights.items()
            if stage.value < self._current_stage.value
        )
        current_weight = self._stage_weights[self._current_stage] * (self._progress / 100)
        return (completed_weight + current_weight) * 100

    @staticmethod
    def _create_progress_bar(progress: float, width: int = 30) -> str:
        """Create a text-based progress bar"""
        filled = int(width * progress / 100)
        return f"{'='*filled}{'-'*(width-filled)}"

class OllamaManager:
    """Manager class for Ollama-related operations"""
    @staticmethod
    def check_ollama(llm_provider: str) -> bool:
        """Check if Ollama is installed and running"""
        try:
            response = subprocess.run(
                ['curl', '-s', 'http://localhost:11434/api/tags'],
                capture_output=True,
                text=True
            )
            
            if response.returncode != 0:
                logger.error("Failed to connect to Ollama")
                return False
                
            model_name = {
                "gemma": gemma_model,
                "deepseek": deepseek_model
            }.get(llm_provider, gemma_model)
            
            model_list = json.loads(response.stdout)
            if not any(model['name'] == model_name for model in model_list['models']):
                logger.warning(f"{model_name} not found. Please run: ollama pull {model_name}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking Ollama: {str(e)}")
            return False

class KnowledgeManager:
    """Manager class for knowledge base operations"""
    
    @staticmethod
    def input_manual_knowledge() -> bool:
        """Handle manual input of knowledge"""
        logger.info("Starting manual knowledge input")
        print("\nManual Knowledge Input Mode")
        print("---------------------------")
        
        # Get description
        description = input("\nEnter the description/query for the 3D model: ").strip()
        if not description:
            logger.warning("Empty description provided")
            print("Description cannot be empty.")
            return False
        
        # Get metadata
        print("\nMetadata Input (optional):")
        print("------------------------")
        metadata = {}
        
        # Complexity
        print("\nSelect complexity level:")
        print("1. Simple")
        print("2. Moderate")
        print("3. Complex")
        print("4. Intricate")
        complexity = input("Enter choice (1-4) or press Enter to skip: ").strip()
        if complexity:
            complexity_map = {
                "1": "SIMPLE",
                "2": "MODERATE",
                "3": "COMPLEX",
                "4": "INTRICATE"
            }
            metadata["complexity"] = complexity_map.get(complexity)
        
        # Style
        print("\nSelect style:")
        print("1. Modern")
        print("2. Traditional")
        print("3. Minimalist")
        print("4. Decorative")
        style = input("Enter choice (1-4) or press Enter to skip: ").strip()
        if style:
            style_map = {
                "1": "Modern",
                "2": "Traditional",
                "3": "Minimalist",
                "4": "Decorative"
            }
            metadata["style"] = style_map.get(style)
        
        # Get OpenSCAD code
        print("\nEnter the OpenSCAD code (press Enter twice to finish):")
        print("----------------------------------------------------")
        
        lines = []
        while True:
            line = input()
            if line.strip() == "" and (not lines or lines[-1].strip() == ""):
                break
            lines.append(line)
        
        scad_code = "\n".join(lines[:-1])
        
        if not scad_code.strip():
            logger.warning("Empty OpenSCAD code provided")
            print("OpenSCAD code cannot be empty.")
            return False
        
        try:
            # Validate OpenSCAD code
            if not KnowledgeManager._validate_scad_code(scad_code):
                return False
            
            # Initialize knowledge base
            kb = EnhancedSCADKnowledgeBase()
            
            # Add example with metadata
            if kb.add_example(description, scad_code):
                logger.info("Knowledge successfully saved")
                print("\nKnowledge has been successfully saved!")
                return True
            else:
                logger.error("Failed to save knowledge")
                print("\nFailed to save knowledge.")
                return False
            
        except Exception as e:
            logger.error(f"Error saving knowledge: {str(e)}")
            print(f"\nError saving knowledge: {str(e)}")
            return False
    
    @staticmethod
    def delete_knowledge() -> bool:
        """Handle knowledge deletion"""
        logger.info("Starting knowledge deletion")
        print("\nKnowledge Deletion Mode")
        print("----------------------")
        
        try:
            kb = EnhancedSCADKnowledgeBase()
            
            # Get all examples
            results = kb.collection.get()
            if not results or not results['ids']:
                print("\nNo examples found in the knowledge base.")
                return False
            
            # Display examples
            print("\nAvailable examples:")
            print("------------------")
            for i, (id, doc) in enumerate(zip(results['ids'], results['documents']), 1):
                print(f"\n{i}. ID: {id}")
                print(f"   Description: {doc[:100]}...")
            
            # Get user selection
            while True:
                choice = input("\nEnter the number of the example to delete (or 'q' to quit): ").strip()
                if choice.lower() == 'q':
                    return False
                
                try:
                    index = int(choice) - 1
                    if 0 <= index < len(results['ids']):
                        example_id = results['ids'][index]
                        break
                    else:
                        print("Invalid selection. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number or 'q' to quit.")
            
            # Confirm deletion
            confirm = input(f"\nAre you sure you want to delete example {example_id}? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Deletion cancelled.")
                return False
            
            # Delete the example
            kb.collection.delete(ids=[example_id])
            logger.info(f"Deleted example {example_id}")
            print(f"\nExample {example_id} has been deleted successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting knowledge: {str(e)}")
            print(f"\nError deleting knowledge: {str(e)}")
            return False
    
    @staticmethod
    def _validate_scad_code(code: str) -> bool:
        """Validate OpenSCAD code for basic syntax and structure"""
        if not code or len(code) < GenerationSettings.min_code_length:
            logger.warning("Code too short")
            print("Error: OpenSCAD code is too short")
            return False
        
        if len(code) > GenerationSettings.max_code_length:
            logger.warning("Code too long")
            print("Error: OpenSCAD code exceeds maximum length")
            return False
        
        required_elements = ['(', ')', '{', '}']
        for element in required_elements:
            if element not in code:
                logger.warning(f"Missing {element} in OpenSCAD code")
                print(f"Error: OpenSCAD code is missing {element}")
                return False
        
        # Check for basic OpenSCAD elements
        basic_keywords = ['module', 'function', 'cube', 'sphere', 'cylinder']
        if not any(keyword in code for keyword in basic_keywords):
            logger.warning("No basic OpenSCAD elements found")
            print("Error: Code doesn't contain any basic OpenSCAD elements")
            return False
        
        return True

def main():
    """Main function for the 3D Model Generator"""
    config = ModelGeneratorConfig()
    progress_manager = ProgressManager()
    logger.info("Starting 3D Model Generator")
    print("Welcome to the 3D Model Generator!")
    
    while True:
        print("\nSelect an option:")
        print("1. Generate a 3D object")
        print("2. Input knowledge manually")
        print("3. Delete knowledge")
        print("4. Quit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "4" or choice.lower() in config.quit_words:
            logger.info("User chose to quit")
            print("\nGoodbye!")
            break
            
        elif choice == "2":
            KnowledgeManager.input_manual_knowledge()
            continue
            
        elif choice == "3":
            KnowledgeManager.delete_knowledge()
            continue
            
        elif choice == "1":
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
                        if not OllamaManager.check_ollama(provider):
                            continue
                    
                    logger.info(f"Initializing generator with provider: {provider}")
                    progress_manager.update_stage(GenerationStage.INITIALIZATION, 50, "Initializing generator")
                    generator = OpenSCADGenerator(provider)
                    progress_manager.update_stage(GenerationStage.INITIALIZATION, 100, "Generator initialized")
                    break
                    
                except Exception as e:
                    error_info = ErrorHandler.handle_generation_error(e, 1, 1)
                    print(f"Error: {error_info['error']}")
                    print("Please check your environment variables and try again.")
                    return
            
            print("\nDescribe the 3D object you want to create, and I'll generate OpenSCAD code for it.")
            print("Type 'quit' to exit.")
            
            while True:
                description = input("\nWhat would you like to model? ")
                if description.lower() in config.quit_words:
                    break
                
                print("I am generating, please be patient...")
                logger.info(f"Generating model for: {description}")
                
                try:
                    for attempt in range(1, config.settings.max_generation_attempts + 1):
                        progress_manager.update_stage(GenerationStage.CODE_GENERATION, 
                                                   (attempt - 1) * 100 / config.settings.max_generation_attempts,
                                                   f"Generation attempt {attempt}")
                        
                        result = generator.generate_model(description)
                        if result['success']:
                            progress_manager.update_stage(GenerationStage.CODE_GENERATION, 100, "Generation complete")
                            break
                            
                        error_info = ErrorHandler.handle_generation_error(
                            Exception(result['error']), 
                            attempt, 
                            config.settings.max_generation_attempts
                        )
                        print(f"\nAttempt {attempt} failed: {error_info['error']}")
                        
                        if attempt == config.settings.max_generation_attempts:
                            print("\nFailed to generate after maximum attempts.")
                            
                except Exception as e:
                    error_info = ErrorHandler.handle_generation_error(
                        e, 
                        config.settings.max_generation_attempts, 
                        config.settings.max_generation_attempts
                    )
                    print(f"\nError: {error_info['error']}")
        
        else:
            logger.warning(f"Invalid choice: {choice}")
            print("Invalid choice. Please select 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()