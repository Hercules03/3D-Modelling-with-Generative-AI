import os
import logging
from typing import Dict, Optional, List, Any
import subprocess
import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import traceback
import datetime
import time

from myAPI import *
from langchain_community.vectorstores import Chroma
from openpyscad import *
from constant import *
from enhanced_scad_knowledge_base import EnhancedSCADKnowledgeBase
from OpenSCAD_Generator import OpenSCADGenerator
from LLMmodel import *
from llm_prompt_logger import LLMPromptLogger
from prompts import STEP_BACK_PROMPT_TEMPLATE

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log', mode='w')  # Only file handler, no StreamHandler
    ]
)

# Add file handler for detailed debug info
debug_handler = logging.FileHandler('detailed_debug.log', mode='w')
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s\n'
    'Function: %(funcName)s\n'
    'Message: %(message)s\n'
    '-------------------\n'
))
logger = logging.getLogger(__name__)
logger.addHandler(debug_handler)

def clear_logs():
    """Clear both debug log files"""
    open('debug.log', 'w').close()
    open('detailed_debug.log', 'w').close()
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
        self._stage_start_times = {}
        self._stage_durations = {}

    def update_stage(self, stage: GenerationStage, progress: float, message: str):
        """Update progress for the current stage"""
        # Record stage transition
        if stage != self._current_stage:
            if self._current_stage:
                duration = time.time() - self._stage_start_times[self._current_stage]
                self._stage_durations[self._current_stage] = duration
                logger.debug(
                    f"Stage Complete: {self._current_stage.value}"
                    f"\n  Duration: {duration:.2f}s"
                    f"\n  Final Progress: {self._progress:.1f}%"
                )
            
            self._stage_start_times[stage] = time.time()
            logger.debug(
                f"Stage Start: {stage.value}"
                f"\n  Time: {datetime.datetime.now().isoformat()}"
                f"\n  Initial Message: {message}"
            )
        
        self._current_stage = stage
        self._progress = progress
        total_progress = self._calculate_total_progress()
        
        # Log detailed progress information to file only
        logger.debug(
            f"Progress Update:"
            f"\n  Stage: {stage.value}"
            f"\n  Stage Progress: {progress:.1f}%"
            f"\n  Total Progress: {total_progress:.1f}%"
            f"\n  Message: {message}"
            f"\n  Time: {datetime.datetime.now().isoformat()}"
        )
        
        # Only show progress bar in terminal
        print(f"\r[{self._create_progress_bar(total_progress)}] {total_progress:.1f}%", end="")

    def _calculate_total_progress(self) -> float:
        """Calculate total progress across all stages"""
        if not self._current_stage:
            return 0.0
        
        completed_weight = sum(
            weight for stage, weight in self._stage_weights.items()
            if stage.value < self._current_stage.value
        )
        current_weight = self._stage_weights[self._current_stage] * (self._progress / 100)
        total_progress = (completed_weight + current_weight) * 100
        
        # Log calculation details
        logger.debug(
            f"Progress Calculation:"
            f"\n  Completed Weight: {completed_weight}"
            f"\n  Current Stage Weight: {self._stage_weights[self._current_stage]}"
            f"\n  Current Progress: {self._progress:.1f}%"
            f"\n  Total Progress: {total_progress:.1f}%"
        )
        
        return total_progress

    def get_stage_summary(self) -> str:
        """Get a summary of all stage durations"""
        summary = ["Stage Duration Summary:"]
        total_duration = 0
        
        for stage, duration in self._stage_durations.items():
            total_duration += duration
            summary.append(f"  {stage.value}: {duration:.2f}s")
        
        if total_duration > 0:
            summary.append(f"\nTotal Duration: {total_duration:.2f}s")
            
        return "\n".join(summary)

    @staticmethod
    def _create_progress_bar(progress: float, width: int = 30) -> str:
        """Create a text-based progress bar"""
        filled = int(width * progress / 100)
        bar = f"{'='*filled}{'-'*(width-filled)}"
        
        # Log progress bar creation
        logger.debug(
            f"Progress Bar:"
            f"\n  Progress: {progress:.1f}%"
            f"\n  Width: {width}"
            f"\n  Filled: {filled}"
            f"\n  Bar: {bar}"
        )
        
        return bar

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

class LLMPromptLogger:
    """Logger for LLM prompt-response pairs"""
    def __init__(self):
        self.log_dir = "conversation_logs"
        self.metadata_file = os.path.join(self.log_dir, "metadata_extraction.json")
        self.category_file = os.path.join(self.log_dir, "category_analysis.json")
        self._init_log_files()

    def _init_log_files(self):
        """Initialize log files if they don't exist"""
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize each file with empty array if it doesn't exist
        for file_path in [self.metadata_file, self.category_file]:
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    json.dump([], f, indent=2)

    def log_metadata_extraction(self, query: str, code: str, response: dict, timestamp: str = None):
        """Log metadata extraction prompt-response pair"""
        if timestamp is None:
            timestamp = datetime.datetime.now().isoformat()
            
        entry = {
            "timestamp": timestamp,
            "input": {
                "description": query,
                "code": code
            },
            "output": response,
            "tokens": {
                "input_tokens": len(query.split()) + len(code.split()),
                "output_tokens": sum(len(str(v).split()) for v in response.values())
            }
        }
        
        self._append_to_json(self.metadata_file, entry)
        logger.debug("Logged metadata extraction: %s", json.dumps(entry, indent=2))

    def log_category_analysis(self, query: str, code: str, response: dict, timestamp: str = None):
        """Log category analysis prompt-response pair"""
        if timestamp is None:
            timestamp = datetime.datetime.now().isoformat()
            
        entry = {
            "timestamp": timestamp,
            "input": {
                "description": query,
                "code": code
            },
            "output": response,
            "tokens": {
                "input_tokens": len(query.split()) + len(code.split()),
                "output_tokens": sum(len(str(v).split()) for v in response.values())
            }
        }
        
        self._append_to_json(self.category_file, entry)
        logger.debug("Logged category analysis: %s", json.dumps(entry, indent=2))

    def _append_to_json(self, file_path: str, new_data: dict):
        """Append new data to existing JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            data.append(new_data)
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error("Error logging to %s: %s", os.path.basename(file_path), str(e))

class KnowledgeManager:
    """Manager class for knowledge base operations"""
    
    _prompt_logger = LLMPromptLogger()  # Class-level instance for prompt logging
    
    @staticmethod
    def input_manual_knowledge() -> bool:
        """Handle manual input of knowledge"""
        logger.info("Starting manual knowledge input")
        logger.debug("Initializing manual knowledge input process")
        print("\nManual Knowledge Input Mode")
        print("---------------------------")
        
        # Get description
        description = input("\nEnter the description/query for the 3D model: ").strip()
        if not description:
            logger.warning("Empty description provided")
            print("Description cannot be empty.")
            return False
            
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
            logger.debug("Starting OpenSCAD code validation")
            if not KnowledgeManager._validate_scad_code(scad_code):
                logger.warning("OpenSCAD code validation failed")
                return False
            
            # Initialize knowledge base
            logger.debug("Initializing knowledge base")
            kb = EnhancedSCADKnowledgeBase()
            
            print("\nExtracting metadata and performing analysis...")
            
            # Extract metadata
            metadata_result = kb.metadata_extractor.extract_metadata(description, scad_code)
            if not metadata_result:
                logger.error("Failed to extract metadata")
                print("Failed to extract metadata from the description.")
                return False
                
            # Log metadata extraction
            KnowledgeManager._prompt_logger.log_metadata_extraction(
                query=description,
                code=scad_code,
                response=metadata_result
            )
            
            # Perform category analysis
            category_result = kb.analyze_categories(description, scad_code)
            if not category_result:
                logger.error("Failed to analyze categories")
                print("Failed to analyze categories.")
                return False
                
            # Log category analysis
            KnowledgeManager._prompt_logger.log_category_analysis(
                query=description,
                code=scad_code,
                response=category_result
            )
            
            # Perform step-back analysis
            step_back_prompt = STEP_BACK_PROMPT_TEMPLATE.format(query=description)
            step_back_response = kb.llm.invoke(step_back_prompt)
            technical_analysis = step_back_response.content
            
            # Parse step-back analysis
            principles = []
            abstractions = []
            approach = []
            
            current_section = None
            for line in technical_analysis.split('\n'):
                line = line.strip()
                if 'CORE PRINCIPLES:' in line:
                    current_section = 'principles'
                elif 'SHAPE COMPONENTS:' in line:
                    current_section = 'abstractions'
                elif 'IMPLEMENTATION STEPS:' in line:
                    current_section = 'approach'
                elif line and line.startswith('-'):
                    if current_section == 'principles':
                        principles.append(line[1:].strip())
                    elif current_section == 'abstractions':
                        abstractions.append(line[1:].strip())
                elif line and line[0].isdigit() and current_section == 'approach':
                    approach.append(line[line.find('.')+1:].strip())
            
            # Add step-back analysis to metadata
            metadata_result['step_back_analysis'] = {
                'principles': principles,
                'abstractions': abstractions,
                'approach': approach
            }
            
            # Analyze required components
            components_prompt = f"""Analyze the following request and identify the key OpenSCAD components needed:
            {description}
            {technical_analysis}
            
            List only the core components needed (modules, primitives, transformations, boolean operations).
            Respond with a valid JSON array of objects, each with 'type' and 'name' fields."""
            
            components_response = kb.llm.invoke(components_prompt)
            try:
                # Clean up JSON response
                json_str = components_response.content.strip()
                if json_str.startswith("```json"):
                    json_str = json_str.split("```json")[1]
                if json_str.endswith("```"):
                    json_str = json_str.rsplit("```", 1)[0]
                metadata_result['components'] = json.loads(json_str.strip())
            except:
                metadata_result['components'] = []
            
            # Display extracted information
            print("\nExtracted Information:")
            print("---------------------")
            print(f"Object Type: {metadata_result.get('object_type', 'Unknown')}")
            print(f"Style: {metadata_result.get('style', 'Unknown')}")
            print(f"Complexity: {metadata_result.get('complexity', 'Unknown')}")
            print("\nCategories:", ", ".join(category_result.get('categories', [])))
            print("\nProperties:", json.dumps(category_result.get('properties', {}), indent=2))
            print("\nTechnical Requirements:", ", ".join(metadata_result.get('technical_requirements', [])))
            
            # Ask for confirmation
            confirm = input("\nDo you want to add this example to the knowledge base? (y/n): ").strip().lower()
            if confirm != 'y':
                logger.info("User cancelled adding example")
                print("Example not added.")
                return False
            
            # Add example with all metadata
            logger.debug("Adding example to knowledge base")
            if kb.add_example(description, scad_code, metadata_result):
                logger.info("Knowledge successfully saved")
                print("\nKnowledge has been successfully saved!")
                return True
            else:
                logger.error("Failed to save knowledge")
                print("\nFailed to save knowledge.")
                return False
            
        except Exception as e:
            logger.error("Error saving knowledge: %s", str(e))
            logger.error("Stack trace: %s", traceback.format_exc())
            print(f"\nError saving knowledge: {str(e)}")
            return False

    @staticmethod
    def delete_knowledge() -> bool:
        """Handle knowledge deletion"""
        logger.info("Starting knowledge deletion")
        print("\nKnowledge Deletion Mode")
        print("----------------------")
        
        try:
            logger.debug("Initializing knowledge base for deletion")
            kb = EnhancedSCADKnowledgeBase()
            
            # Get all examples
            logger.debug("Retrieving all examples")
            results = kb.collection.get()
            if not results or not results['ids']:
                logger.info("No examples found in knowledge base")
                print("\nNo examples found in the knowledge base.")
                return False
            
            # Log retrieved examples
            logger.debug("Found %d examples", len(results['ids']))
            
            # Display examples
            print("\nAvailable examples:")
            print("------------------")
            for i, (id, doc) in enumerate(zip(results['ids'], results['documents']), 1):
                print(f"\n{i}. ID: {id}")
                print(f"   Description: {doc[:100]}...")
                logger.debug("Example %d: ID=%s, Description=%s", i, id, doc[:100])
            
            # Get user selection
            while True:
                choice = input("\nEnter the number of the example to delete (or 'q' to quit): ").strip()
                if choice.lower() == 'q':
                    logger.info("User cancelled deletion")
                    return False
                
                try:
                    index = int(choice) - 1
                    if 0 <= index < len(results['ids']):
                        example_id = results['ids'][index]
                        logger.debug("Selected example %d with ID: %s", index + 1, example_id)
                        break
                    else:
                        logger.warning("Invalid selection index: %d", index + 1)
                        print("Invalid selection. Please try again.")
                except ValueError:
                    logger.warning("Invalid input: %s", choice)
                    print("Invalid input. Please enter a number or 'q' to quit.")
            
            # Confirm deletion
            confirm = input(f"\nAre you sure you want to delete example {example_id}? (y/n): ").strip().lower()
            if confirm != 'y':
                logger.info("Deletion cancelled by user")
                print("Deletion cancelled.")
                return False
            
            # Delete the example
            logger.debug("Deleting example with ID: %s", example_id)
            kb.collection.delete(ids=[example_id])
            logger.info("Successfully deleted example %s", example_id)
            print(f"\nExample {example_id} has been deleted successfully!")
            return True
            
        except Exception as e:
            logger.error("Error deleting knowledge: %s", str(e))
            logger.error("Stack trace: %s", traceback.format_exc())
            print(f"\nError deleting knowledge: {str(e)}")
            return False

    @staticmethod
    def _validate_scad_code(code: str) -> bool:
        """Validate OpenSCAD code for basic syntax and structure"""
        logger.debug("Validating OpenSCAD code")
        logger.debug("Code length: %d characters", len(code))
        
        if not code or len(code) < GenerationSettings.min_code_length:
            logger.warning("Code too short: %d characters (minimum: %d)", 
                         len(code), GenerationSettings.min_code_length)
            print("Error: OpenSCAD code is too short")
            return False
        
        if len(code) > GenerationSettings.max_code_length:
            logger.warning("Code too long: %d characters (maximum: %d)", 
                         len(code), GenerationSettings.max_code_length)
            print("Error: OpenSCAD code exceeds maximum length")
            return False
        
        required_elements = ['(', ')', '{', '}']
        missing_elements = []
        for element in required_elements:
            if element not in code:
                missing_elements.append(element)
                logger.warning(f"Missing {element} in OpenSCAD code")
                print(f"Error: OpenSCAD code is missing {element}")
        
        if missing_elements:
            logger.warning("Missing required elements: %s", ", ".join(missing_elements))
            return False
        
        # Check for basic OpenSCAD elements
        basic_keywords = ['module', 'function', 'cube', 'sphere', 'cylinder']
        found_keywords = [keyword for keyword in basic_keywords if keyword in code]
        
        if not found_keywords:
            logger.warning("No basic OpenSCAD elements found")
            print("Error: Code doesn't contain any basic OpenSCAD elements")
            return False
        
        logger.debug("Found OpenSCAD keywords: %s", ", ".join(found_keywords))
        logger.info("OpenSCAD code validation successful")
        return True

def main():
    """Main function for the 3D Model Generator"""
    config = ModelGeneratorConfig()
    progress_manager = ProgressManager()
    
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
    
    while True:
        print("\nSelect an option:")
        print("1. Generate a 3D object")
        print("2. Input knowledge manually")
        print("3. Delete knowledge")
        print("4. Quit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "4" or choice.lower() in config.quit_words:
            # Log final summary before exiting
            logger.info("\nSession Summary:")
            logger.info(progress_manager.get_stage_summary())
            logger.info("Session ended at: %s", datetime.datetime.now().isoformat())
            logger.info("=" * 50)
            
            print("\nGoodbye!")
            break
            
        elif choice == "2":
            logger.info("Starting manual knowledge input")
            result = KnowledgeManager.input_manual_knowledge()
            logger.info("Manual knowledge input result: %s", "Success" if result else "Failed")
            continue
            
        elif choice == "3":
            logger.info("Starting knowledge deletion")
            result = KnowledgeManager.delete_knowledge()
            logger.info("Knowledge deletion result: %s", "Success" if result else "Failed")
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
                    logger.error("Provider initialization failed: %s", error_info)
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
                            # Log successful generation details (to file only)
                            logger.info("\nGeneration Summary:")
                            logger.info("  Description: %s", description)
                            logger.info("  Provider: %s", provider)
                            logger.info("  Attempts: %d", attempt)
                            logger.info("  Stage Summary:\n%s", progress_manager.get_stage_summary())
                            print("\nGeneration complete!")  # Simple success message for terminal
                            break
                            
                        error_info = ErrorHandler.handle_generation_error(
                            Exception(result['error']), 
                            attempt, 
                            config.settings.max_generation_attempts
                        )
                        print(f"\rAttempt {attempt} failed. Retrying..." if attempt < config.settings.max_generation_attempts else "\nFailed to generate model.")
                        
                        if attempt == config.settings.max_generation_attempts:
                            logger.error("Failed to generate after maximum attempts")
                            logger.error("Stage Summary:\n%s", progress_manager.get_stage_summary())
                            
                except Exception as e:
                    error_info = ErrorHandler.handle_generation_error(
                        e, 
                        config.settings.max_generation_attempts, 
                        config.settings.max_generation_attempts
                    )
                    print("\nAn error occurred. Please check the debug logs for details.")
                    logger.error("Unexpected error during generation: %s", error_info)
                    logger.error("Stage Summary:\n%s", progress_manager.get_stage_summary())
        
        else:
            logger.warning(f"Invalid choice: {choice}")
            print("Invalid choice. Please select 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()