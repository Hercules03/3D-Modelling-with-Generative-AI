import os
import logging
# Set tokenizers parallelism to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Dict, Optional, List, Any
import subprocess
import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import traceback
import datetime
import time

# Import ModelDefinitions first since other modules depend on it
from llm_management import ModelDefinitions

from langchain_community.vectorstores import Chroma
from openpyscad import *
from constant import *
from enhanced_scad_knowledge_base import EnhancedSCADKnowledgeBase
from conversation_logger import ConversationLogger
from OpenSCAD_Generator import OpenSCADGenerator
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
            "1": {"name": "anthropic", "model": ModelDefinitions.ANTHROPIC},
            "2": {"name": "openai", "model": ModelDefinitions.OPENAI},
            "3": {"name": "gemma", "model": ModelDefinitions.GEMMA},
            "4": {"name": "deepseek", "model": ModelDefinitions.DEEPSEEK}
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
                "gemma": ModelDefinitions.GEMMA,
                "deepseek": ModelDefinitions.DEEPSEEK
            }.get(llm_provider, ModelDefinitions.GEMMA)
            
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
    
    """ 
    Workflow
    ================================
    1. Get description
    2. Get OpenSCAD code (add.scad)
    3. Validate OpenSCAD code    # Need check the mechanism of the code
    4. Keyword extraction
    5. Perform step-back analysis
    6. Extract metadata
    7. Perform category analysis
    8. Analyze required components
    9. Add example to knowledge base
    """
    
    _prompt_logger = LLMPromptLogger()  # Class-level instance for prompt logging
    
    @staticmethod
    def input_manual_knowledge() -> bool:
        """Handle manual input of knowledge"""
        logger.info("Starting manual knowledge input")
        logger.debug("Initializing manual knowledge input process")
        print("\nManual Knowledge Input Mode")
        print("---------------------------")
        
        # Step 1: Get description
        description = input("\nEnter the description/query for the 3D model: ").strip()
        if not description:
            logger.warning("Empty description provided")
            print("Description cannot be empty.")
            return False
            
        # Step 2: Get OpenSCAD code from add.scad
        try:
            with open("add.scad", "r") as f:
                scad_code = f.read().strip()
            if not scad_code:
                logger.warning("Empty OpenSCAD code provided")
                print("OpenSCAD code cannot be empty.")
                return False
            print("\nRead OpenSCAD code from add.scad:")
            print("-" * 52)
            print(scad_code)
            print("-" * 52 + "\n")
            
            # Validate OpenSCAD code
            logger.debug("Starting OpenSCAD code validation")
            if not KnowledgeManager._validate_scad_code(scad_code):
                logger.warning("OpenSCAD code validation failed")
                return False
                
        except FileNotFoundError:
            logger.error("add.scad file not found")
            print("Error: add.scad file not found. Please create the file with your OpenSCAD code.")
            return False
        except Exception as e:
            logger.error(f"Error reading add.scad: {str(e)}")
            print(f"Error reading add.scad: {str(e)}")
            return False
        
        try:
            # Initialize knowledge base and logger
            logger.debug("Initializing knowledge base")
            kb = EnhancedSCADKnowledgeBase()
            convLogger = ConversationLogger()
            
            # Step 1: Extract keywords and get user confirmation
            print("\n" + "="*50)
            print("STEP 1: KEYWORD EXTRACTION")
            print("="*50)
            
            keyword_data = None
            max_keyword_retries = 3
            keyword_retry_count = 0
            
            while keyword_retry_count < max_keyword_retries:
                # Log keyword extraction query and prompt
                keyword_prompt = kb.keyword_extractor.prompt.replace("<<INPUT>>", description)
                logger.debug(
                    "=== KEYWORD EXTRACTION ===\n"
                    f"Attempt {keyword_retry_count + 1}/{max_keyword_retries}\n"
                    "Query:\n"
                    f"{description}\n\n"
                    "Full Prompt Sent to LLM:\n"
                    f"{keyword_prompt}\n\n"
                )
                
                keyword_data = kb.keyword_extractor.extract_keyword(description)
                
                # Log keyword extraction response
                logger.debug(
                    "Response:\n"
                    f"Core Type: {keyword_data.get('core_type', '')}\n"
                    f"Modifiers: {', '.join(keyword_data.get('modifiers', []))}\n"
                    f"Compound Type: {keyword_data.get('compound_type', '')}\n"
                    "=" * 50 + "\n\n"
                )
                
                print("\nKeyword Analysis Results:")
                print("-" * 30)
                print(f"Core Type: {keyword_data.get('core_type', '')}")
                print(f"Modifiers: {', '.join(keyword_data.get('modifiers', []))}")
                print(f"Compound Type: {keyword_data.get('compound_type', '')}")
                print("-" * 30)
                
                # Ask for user confirmation
                user_input = input("\nDo you accept these keywords? (yes/no): ").lower().strip()
                
                # Log user's keyword decision
                logger.debug(
                    "=== USER KEYWORD DECISION ===\n"
                    f"User accepted keywords: {user_input == 'yes'}\n"
                    "=" * 50 + "\n\n"
                )
                
                if user_input == 'yes':
                    # Log the approved keywords
                    convLogger.log_keyword_extraction({
                        "query": {
                            "input": description,
                            "timestamp": datetime.datetime.now().isoformat()
                        },
                        "response": keyword_data,
                        "metadata": {
                            "success": True,
                            "error": None,
                            "user_approved": True
                        }
                    })
                    break
                else:
                    keyword_retry_count += 1
                    if keyword_retry_count < max_keyword_retries:
                        print("\nRetrying keyword extraction...")
                        # Ask user for refinement suggestions
                        print("Please provide any suggestions to improve the keyword extraction (or press Enter to retry):")
                        user_feedback = input().strip()
                        if user_feedback:
                            description = f"{description}\nConsider these adjustments: {user_feedback}"
                    else:
                        print("\nMaximum keyword extraction attempts reached.")
                        print("Please try again with a different description.")
                        return False

            if keyword_data is None:
                print("\nKeyword extraction failed. Please try again with a different description.")
                return False

            # Step 2: Perform step-back analysis with approved keywords
            print("\n" + "="*50)
            print("STEP 2: TECHNICAL ANALYSIS")
            print("="*50)
            
            step_back_result = None
            max_step_back_retries = 3
            step_back_retry_count = 0
            
            while step_back_retry_count < max_step_back_retries:
                # Log step-back analysis query and prompt
                step_back_prompt = STEP_BACK_PROMPT_TEMPLATE.format(
                    Object=keyword_data.get('compound_type', '') or keyword_data.get('core_type', ''),
                    Type=keyword_data.get('core_type', ''),
                    Modifiers=', '.join(keyword_data.get('modifiers', []))
                )
                
                logger.debug(
                    "=== STEP-BACK ANALYSIS ===\n"
                    f"Attempt {step_back_retry_count + 1}/{max_step_back_retries}\n"
                    "Query:\n"
                    f"{description}\n\n"
                    "Keyword Data:\n"
                    f"Core Type: {keyword_data.get('core_type', '')}\n"
                    f"Modifiers: {', '.join(keyword_data.get('modifiers', []))}\n"
                    f"Compound Type: {keyword_data.get('compound_type', '')}\n\n"
                    "Full Prompt Sent to LLM:\n"
                    f"{step_back_prompt}\n\n"
                )
                
                step_back_result = kb.perform_step_back(description, keyword_data)
                if not step_back_result:
                    msg = "Step-back analysis failed. Proceeding with basic generation..."
                    print(msg)
                    logger.debug(f"{msg}\n\n")
                    return False
                
                # Log step-back analysis results
                logger.debug(
                    "Response:\n"
                    "Core Principles:\n"
                    "\n".join(f"- {p}" for p in step_back_result.get('principles', [])) + "\n\n"
                    "Shape Components:\n"
                    "\n".join(f"- {a}" for a in step_back_result.get('abstractions', [])) + "\n\n"
                    "Implementation Steps:\n"
                    "\n".join(f"{i+1}. {s}" for i, s in enumerate(step_back_result.get('approach', []))) + "\n"
                    "=" * 50 + "\n\n"
                )
                
                """
                print("\nStep-back Analysis Results:")
                print("-" * 30)
                print("Core Principles:")
                for p in step_back_result.get('principles', []):
                    print(f"- {p}")
                print("\nShape Components:")
                for a in step_back_result.get('abstractions', []):
                    print(f"- {a}")
                print("\nImplementation Steps:")
                for i, s in enumerate(step_back_result.get('approach', []), 1):
                    print(f"{i}. {s}")
                print("-" * 30)
                """
                
                # Ask for user confirmation
                user_input = input("\nDo you accept this technical analysis? (yes/no): ").lower().strip()
                
                # Log user's step-back decision
                logger.debug(
                    "=== USER STEP-BACK DECISION ===\n"
                    f"User accepted step-back analysis: {user_input == 'yes'}\n"
                    "=" * 50 + "\n\n"
                )
                
                if user_input == 'yes':
                    # Log the approved step-back analysis
                    convLogger.log_step_back_analysis({
                        "query": {
                            "input": description,
                            "timestamp": datetime.datetime.now().isoformat()
                        },
                        "response": {
                            "principles": step_back_result.get('principles', []),
                            "abstractions": step_back_result.get('abstractions', []),
                            "approach": step_back_result.get('approach', [])
                        },
                        "metadata": {
                            "success": True,
                            "error": None,
                            "user_approved": True
                        }
                    })
                    break
                else:
                    step_back_retry_count += 1
                    if step_back_retry_count < max_step_back_retries:
                        print("\nRetrying step-back analysis...")
                        # Ask user for refinement suggestions
                        print("Please provide any suggestions to improve the step-back analysis (or press Enter to retry):")
                        user_feedback = input().strip()
                        if user_feedback:
                            description = f"{description}\nConsider these aspects in your analysis: {user_feedback}"
                    else:
                        print("\nMaximum step-back analysis attempts reached.")
                        print("Please try again with a different description.")
                        return False

            if step_back_result is None:
                print("\nStep-back analysis failed. Please try again with a different description.")
                return False

            # Step 3: Add example to knowledge base
            print("\n" + "="*50)
            print("STEP 3: ADDING TO KNOWLEDGE BASE")
            print("="*50)
            
            # Get relevant examples for similarity comparison
            examples, metadata = kb.get_similar_examples(description, return_metadata=True)
            
            # Prepare metadata for the new example
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
            example_id = f"manual_{formatted_time}"
            
            example_metadata = {
                "id": example_id,
                "object_type": keyword_data.get('core_type', ''),
                "features": keyword_data.get('modifiers', []),
                "step_back_analysis": step_back_result,
                "timestamp": current_time.isoformat(),
                "user_accepted": True,
                "type": "manual"
            }
            
            # Add example to knowledge base
            try:
                success = kb.add_example(description, scad_code, example_metadata)
                if success:
                    print(f"\nExample successfully added to knowledge base with ID: {example_id}")
                    logger.debug(
                        "=== EXAMPLE ADDED TO KNOWLEDGE BASE ===\n"
                        f"ID: {example_id}\n"
                        f"Description: {description}\n"
                        f"Code length: {len(scad_code)} characters\n"
                        "Metadata:\n"
                        f"Object Type: {example_metadata['object_type']}\n"
                        f"Features: {', '.join(example_metadata['features'])}\n"
                        f"Timestamp: {example_metadata['timestamp']}\n"
                        "=" * 50 + "\n\n"
                    )
                    return True
                else:
                    print("\nFailed to add example to knowledge base.")
                    logger.debug(
                        "=== FAILED TO ADD EXAMPLE ===\n"
                        "add_example() returned False\n"
                        "=" * 50 + "\n\n"
                    )
                    return False
            except Exception as e:
                print(f"\nError adding example to knowledge base: {str(e)}")
                logger.debug(
                    "=== ERROR ADDING EXAMPLE ===\n"
                    f"Exception: {str(e)}\n"
                    f"Traceback:\n{traceback.format_exc()}\n"
                    "=" * 50 + "\n\n"
                )
                return False
                
        except Exception as e:
            logger.error("Error saving knowledge: %s", str(e))
            logger.error("Stack trace: %s", traceback.format_exc())
            print(f"\nError saving knowledge: {str(e)}")
            return False

    @staticmethod
    def delete_knowledge() -> bool:
        """Handle knowledge deletion with improved user experience"""
        logger.info("Starting knowledge deletion")
        print("\nKnowledge Deletion Mode")
        print("----------------------")
        
        try:
            logger.debug("Initializing knowledge base for deletion")
            kb = EnhancedSCADKnowledgeBase()
            
            while True:
                # Show menu
                print("\nDelete Knowledge Options:")
                print("1. Search by description")
                print("2. Filter by metadata")
                print("3. View all examples")
                print("4. Return to main menu")
                
                choice = input("\nEnter your choice (1-4): ").strip()
                
                if choice == "4":
                    logger.info("User exited deletion mode")
                    return True
                    
                # Initialize search parameters
                search_term = None
                filters = {}
                page = 1
                page_size = 5
                
                if choice == "1":
                    search_term = input("\nEnter search term: ").strip()
                elif choice == "2":
                    print("\nAvailable filters:")
                    print("1. Style (e.g., Modern, Traditional, Minimalist)")
                    print("2. Complexity (SIMPLE, MEDIUM, COMPLEX)")
                    print("3. Object Type")
                    
                    filter_choice = input("\nEnter filter number (1-3): ").strip()
                    
                    if filter_choice == "1":
                        style = input("Enter style: ").strip()
                        filters["style"] = style
                    elif filter_choice == "2":
                        complexity = input("Enter complexity: ").strip().upper()
                        filters["complexity"] = complexity
                    elif filter_choice == "3":
                        obj_type = input("Enter object type: ").strip()
                        filters["object_type"] = obj_type
                
                # Search for examples
                while True:
                    results = kb.search_examples(search_term, filters, page, page_size)
                    
                    if not results['examples']:
                        print("\nNo examples found.")
                        break
                    
                    print(f"\nShowing page {results['page']} of {results['total_pages']} (Total: {results['total']} examples)")
                    print("\nAvailable examples:")
                    print("------------------")
                    
                    for i, example in enumerate(results['examples'], 1):
                        print(f"\n{i}. ID: {example['id']}")
                        print(f"   Description: {example['description'][:100]}...")
                        print("   Metadata:")
                        print(f"   - Style: {example['metadata'].get('style', 'N/A')}")
                        print(f"   - Complexity: {example['metadata'].get('complexity', 'N/A')}")
                        print(f"   - Object Type: {example['metadata'].get('object_type', 'N/A')}")
                    
                    # Show navigation options
                    print("\nOptions:")
                    print("1-5: Select example to delete")
                    print("n: Next page")
                    print("p: Previous page")
                    print("b: Back to delete menu")
                    print("q: Return to main menu")
                    
                    action = input("\nEnter your choice: ").strip().lower()
                    
                    if action == 'q':
                        logger.info("User exited deletion mode")
                        return True
                    elif action == 'b':
                        break
                    elif action == 'n' and page < results['total_pages']:
                        page += 1
                    elif action == 'p' and page > 1:
                        page -= 1
                    elif action.isdigit():
                        index = int(action) - 1
                        if 0 <= index < len(results['examples']):
                            example = results['examples'][index]
                            
                            # Show example details
                            print("\nExample Details:")
                            print("-" * 40)
                            print(f"ID: {example['id']}")
                            print(f"Description: {example['description']}")
                            print("\nMetadata:")
                            for key, value in example['metadata'].items():
                                if key != 'code':  # Skip code for cleaner display
                                    print(f"{key}: {value}")
            
            # Confirm deletion
                            confirm = input(f"\nAre you sure you want to delete this example? (y/n): ").strip().lower()
                            if confirm == 'y':
                                if kb.delete_examples([example['id']]):
                                    print(f"\nExample {example['id']} has been deleted successfully!")
                                    logger.info(f"Deleted example {example['id']}")
                                    # Refresh the current page
                                    results = kb.search_examples(search_term, filters, page, page_size)
                                    if not results['examples'] and page > 1:
                                        page -= 1
                                else:
                                    print("\nFailed to delete example.")
                                    logger.error(f"Failed to delete example {example['id']}")
                        else:
                            print("\nInvalid selection.")
                
            return True
            
        except Exception as e:
            logger.error("Error in delete_knowledge: %s", str(e))
            logger.error("Stack trace: %s", traceback.format_exc())
            print(f"\nError in delete_knowledge: {str(e)}")
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
        print("4. View knowledge base")
        print("5. Quit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "5" or choice.lower() in config.quit_words:
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
            
        elif choice == "4":
            logger.info("Starting knowledge base viewer")
            try:
                generator = OpenSCADGenerator()
                generator.view_knowledge_base()
                logger.info("Knowledge base viewing completed")
            except Exception as e:
                error_msg = f"Error viewing knowledge base: {str(e)}"
                logger.error(error_msg)
                print(f"\nError: {error_msg}")
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
            print("Invalid choice. Please select 1, 2, 3, 4, or 5.")

if __name__ == "__main__":
    main()