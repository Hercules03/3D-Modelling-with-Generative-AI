from prompts import OPENSCAD_GNERATOR_PROMPT_TEMPLATE, BASIC_KNOWLEDGE
from constant import *
from KeywordExtractor import KeywordExtractor
from LLM import LLMProvider
from scad_knowledge_base import SCADKnowledgeBase
from step_back_analyzer import StepBackAnalyzer
from langchain.prompts import ChatPromptTemplate
import datetime
from typing import Optional, List, Dict
import logging
import traceback

logger = logging.getLogger(__name__)

class OpenSCADGenerator:
    def __init__(self, llm_provider="anthropic", knowledge_base=None, keyword_extractor=None, metadata_extractor=None, conversation_logger=None):
        """Initialize the OpenSCAD generator"""
        print("\n=== Initializing OpenSCAD Generator ===")
        logger.info("Initializing OpenSCAD Generator...")
        # Initialize LLM
        print("\nSetting up LLM...")
        logger.info("Setting up LLM...")
        self.llm_provider = llm_provider
        self.llm = LLMProvider.get_llm(provider=llm_provider)
        self.model_name = self.llm.model_name if hasattr(self.llm, 'model_name') else str(self.llm)
        print(f"- Provider: {self.llm_provider}")
        print(f"- Model: {self.model_name}")
        logger.info(f"- Provider: {self.llm_provider}")
        logger.info(f"- Model: {self.model_name}")
    
        # Initialize knowledge base and logger
        print("\nSetting up components...")
        logger.info("Setting up components...")
        self.knowledge_base = knowledge_base
        self.logger = conversation_logger
        self.keyword_extractor = keyword_extractor
        self.metadata_extractor = metadata_extractor
        
        # Initialize step-back analyzer
        self.step_back_analyzer = StepBackAnalyzer(llm=self.llm, conversation_logger=self.logger)
        
        # Load prompts
        print("\nLoading prompts...")
        logger.info("Loading OpenSCAD Generator prompts...")
        self.main_prompt = OPENSCAD_GNERATOR_PROMPT_TEMPLATE
        print("- Main generation prompt loaded")
        logger.info("- Main generation prompt loaded")
        
        # Initialize debug log
        self.debug_log = []
        print("\n=== OpenSCAD Generator Ready ===\n")
        
    def write_debug(self, *messages):
        """Write messages to debug log"""
        for message in messages:
            self.debug_log.append(message)
            
    def save_debug_log(self):
        """Save debug log to file"""
        try:
            with open("debug_log.txt", "w") as f:
                f.write("".join(self.debug_log))
        except Exception as e:
            print(f"Error saving debug log: {e}")
    
    def perform_step_back_analysis(self, description: str, keyword_data: dict) -> Optional[dict]:
        """Perform step-back analysis with approved keywords.
        
        Args:
            description: The description to analyze
            keyword_data: The extracted keyword data
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary containing step-back analysis if successful, None otherwise
        """
        return self.step_back_analyzer.perform_analysis(description, keyword_data)

    def perform_keyword_extraction(self, description: str, max_retries: int = 3) -> Optional[dict]:
        """Perform keyword extraction and get user confirmation.
        
        Args:
            description: The description to extract keywords from
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary containing keyword data if successful, None otherwise
        """
        keyword_data = None
        retry_count = 0
        print(description)
        
        while retry_count < max_retries:
            # Log keyword extraction query and prompt
            self.write_debug(
                "=== KEYWORD EXTRACTION ===\n",
                f"Attempt {retry_count + 1}/{max_retries}\n",
                "Query:\n",
                f"{description}\n\n",
                "Extracting keywords from description...\n\n"
            )
            
            keyword_data = self.keyword_extractor.extract_keyword(description)
            
            # Log keyword extraction response
            self.write_debug(
                "Response:\n",
                f"Core Type: {keyword_data.get('core_type', '')}\n",
                f"Modifiers: {', '.join(keyword_data.get('modifiers', []))}\n",
                f"Compound Type: {keyword_data.get('compound_type', '')}\n",
                "=" * 50 + "\n\n"
            )
            
            
            print("\nKeyword Analysis Results:")
            print("-" * 30)
            print(f"query: {description}")
            print(f"Core Type: {keyword_data.get('core_type', '')}")
            print(f"Modifiers: {', '.join(keyword_data.get('modifiers', []))}")
            print(f"Compound Type: {keyword_data.get('compound_type', '')}")
            print("-" * 30)
            
            # Ask for user confirmation
            user_input = input("\nDo you accept these keywords? (yes/no): ").lower().strip()
            
            # Log user's keyword decision
            self.write_debug(
                "=== USER KEYWORD DECISION ===\n",
                f"User accepted keywords: {user_input == 'yes'}\n",
                "=" * 50 + "\n\n"
            )
            
            if user_input == 'yes':
                # Log the approved keywords
                self.logger.log_keyword_extraction({
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
                return keyword_data
            
            retry_count += 1
            if retry_count < max_retries:
                print("\nRetrying keyword extraction...")
                # Ask user for refinement suggestions
                print("Please provide any suggestions to improve the keyword extraction (or press Enter to retry):")
                user_feedback = input().strip()
                if user_feedback:
                    description = f"{description}\nConsider these adjustments: {user_feedback}"
            else:
                print("\nMaximum keyword extraction attempts reached.")
                print("Please try again with a different description.")
        
        return None

    def retrieve_similar_examples(self, description: str, step_back_result: dict, keyword_data: dict) -> tuple[list, Optional[dict]]:
        """Retrieve similar examples from the knowledge base."""
        print("\nRetrieving relevant examples...")
        return self.knowledge_base.get_examples(
            description,
            step_back_result=step_back_result,
            keyword_data=keyword_data,
            return_metadata=True
        )

    def prepare_generation_inputs(self, description: str, examples: List[Dict], step_back_result: Dict = None) -> Dict:
        """
        Prepare inputs for the code generation prompt.
        
        Args:
            description: The original query/description
            examples: List of similar examples found
            step_back_result: Optional step-back analysis results
            
        Returns:
            Dictionary containing all inputs needed for the generation prompt
        """
        try:
            # Format step-back analysis if available
            step_back_text = ""
            if step_back_result:
                principles = step_back_result.get('principles', [])
                abstractions = step_back_result.get('abstractions', [])
                approach = step_back_result.get('approach', [])
                
                step_back_text = f"""
                CORE PRINCIPLES:
                {chr(10).join(f'- {p}' for p in principles)}
                
                SHAPE COMPONENTS:
                {chr(10).join(f'- {a}' for a in abstractions)}
                
                IMPLEMENTATION STEPS:
                {chr(10).join(f'{i+1}. {s}' for i, s in enumerate(approach))}
                """
            
            # Format examples for logging
            examples_text = []
            for ex in examples:
                example_id = ex.get('example', {}).get('id', 'unknown')
                score = ex.get('score', 0.0)
                score_breakdown = ex.get('score_breakdown', {})
                
                example_text = f"""
                Example ID: {example_id}
                Score: {score:.3f}
                Component Scores:
                {chr(10).join(f'  - {name}: {score:.3f}' for name, score in score_breakdown.get('component_scores', {}).items())}
                """
                examples_text.append(example_text)
            
            # Prepare the complete inputs
            inputs = {
                "basic_knowledge": BASIC_KNOWLEDGE,
                "examples": examples,
                "request": description,
                "step_back_analysis": step_back_text.strip() if step_back_text else ""
            }
            
            # Log the complete analysis and examples
            """
            print("\nStep-back Analysis:")
            print(step_back_text.strip() if step_back_text else "No step-back analysis available")
            """
            
            print("\nRetrieved Examples:")
            if examples_text:
                print("\n".join(examples_text))
            else:
                print("No examples found")
            
            return inputs
            
        except Exception as e:
            print(f"Error preparing generation inputs: {str(e)}")
            traceback.print_exc()
            return {
                "basic_knowledge": BASIC_KNOWLEDGE,
                "examples": [],
                "request": description,
                "step_back_analysis": ""
            }

    def generate_scad_code(self, description: str, examples: list, step_back_result: dict) -> Optional[dict]:
        """Generate OpenSCAD code using the prepared inputs.
        
        Args:
            description: The description to generate code for
            examples: List of similar examples
            step_back_result: The step-back analysis results
            
        Returns:
            Dictionary containing success status and generated code/error
        """
        # Prepare the prompt inputs
        inputs = self.prepare_generation_inputs(
            description=description,
            examples=examples,
            step_back_result=step_back_result
        )
        
        # Generate the prompt and log it
        prompt_value = self.main_prompt.format(**inputs)
        
        # Log SCAD generation query with full prompt
        self.write_debug(
            "=== SCAD CODE GENERATION ===\n",
            "Query:\n",
            f"Description: {description}\n",
            f"Number of Examples Used: {len(examples)}\n",
            "Step-back Analysis Used: Yes\n\n",
            "Full Prompt Sent to LLM:\n",
            f"{prompt_value}\n\n"
        )
        
        # Get response from the LLM
        print("Generating OpenSCAD code...")
        print("Thinking...", end="", flush=True)
        
        # Get streaming response
        content = ""
        for chunk in self.llm.stream(prompt_value):
            if hasattr(chunk, 'content'):
                chunk_content = chunk.content
            else:
                chunk_content = str(chunk)
            content += chunk_content
            print(".", end="", flush=True)
        
        print("\n")  # New line after progress dots
        
        # Log SCAD generation response
        self.write_debug(
            "Response:\n",
            f"{content}\n\n",
            "=" * 50 + "\n\n"
        )
        
        # Try to extract code with different tag variations
        code_tags = [
            ('<code>', '</code>'),
            ('```scad', '```'),
            ('```openscad', '```'),
            ('```', '```')
        ]
        
        scad_code = None
        for start_tag, end_tag in code_tags:
            if start_tag in content and end_tag in content:
                code_start = content.find(start_tag) + len(start_tag)
                code_end = content.find(end_tag, code_start)
                if code_end > code_start:
                    scad_code = content[code_start:code_end].strip()
                    break
        
        if not scad_code:
            return {
                'success': False,
                'error': "No code section found in response"
            }
        
        # Add generated code to debug log
        self.write_debug(
            "=== GENERATED SCAD CODE ===\n",
            f"{scad_code}\n",
            "=" * 50 + "\n\n"
        )
        
        return {
            'success': True,
            'code': scad_code,
            'prompt': prompt_value
        }

    def generate_model(self, description):
        """Generate OpenSCAD code for the given description and save it to a file."""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Clear previous debug log if this is first attempt
                if retry_count == 0:
                    logger.info("First attempt, clearing previous debug log...")
                    self.debug_log = []
                    
                    # Log initial user decision and query
                    self.write_debug(
                        "=== USER DECISION AND INITIAL QUERY ===\n",
                        "Decision: User chose to generate a new 3D model\n",
                        f"Model Provider: {self.llm_provider}\n",
                        f"Model Name: {self.model_name}\n",
                        f"Raw Query: {description}\n",
                        "=" * 50 + "\n\n"
                    )
                
                # Step 1: Extract keywords and get user confirmation
                print("\n" + "="*50)
                print("STEP 1: KEYWORD EXTRACTION")
                print("="*50)
                
                keyword_data = self.perform_keyword_extraction(description)
                if keyword_data is None:
                    print("\nKeyword extraction failed. Please try again with a different description.")
                    return None

                # Step 2: Perform step-back analysis with approved keywords
                print("\n" + "="*50)
                print("STEP 2: TECHNICAL ANALYSIS")
                print("="*50)
                
                step_back_result = self.perform_step_back_analysis(description, keyword_data)
                if step_back_result is None:
                    print("\nStep-back analysis failed. Please try again with a different description.")
                    return None
                
                # Step 3: Get relevant examples from knowledge base
                print("\n" + "="*50)
                print("STEP 3: FINDING SIMILAR EXAMPLES")
                print("="*50)
                
                examples = []
                extracted_metadata = None
                if retry_count == 0:  # Only get examples on first try
                    examples, extracted_metadata = self.retrieve_similar_examples(
                        description, step_back_result, keyword_data
                    )
                
                # Step 4: Generate OpenSCAD code
                print("\n" + "="*50)
                print("STEP 4: GENERATING SCAD CODE")
                print("="*50)
                
                result = self.generate_scad_code(description, examples, step_back_result)
                if not result['success']:
                    error_msg = result['error']
                    print(f"\nError: {error_msg}")
                    self.write_debug(
                        "=== GENERATION ERROR ===\n",
                        f"Error: {error_msg}\n",
                        "=" * 50 + "\n\n"
                    )
                    
                    # Increment retry counter and continue if not max retries
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"Retrying... ({retry_count}/{max_retries})")
                        continue
                    else:
                        self.save_debug_log()
                        return {
                            'success': False,
                            'error': error_msg
                        }
                
                scad_code = result['code']
                prompt_value = result['prompt']
                
                # Save to file
                with open("output.scad", "w") as f:
                    f.write(scad_code)
                
                print("\nOpenSCAD code has been generated and saved to 'output.scad'")
                print("\nGenerated Code:")
                print("-" * 40)
                print(scad_code)
                print("-" * 40)
                
                # Ask user if they want to add this to the knowledge base
                add_to_kb = input("\nWould you like to add this example to the knowledge base? (y/n): ").lower().strip()
                
                # Add user decision to debug log
                self.write_debug(
                    "=== USER DECISION ===\n",
                    f"Add to knowledge base: {add_to_kb}\n\n"
                )
                
                if add_to_kb == 'y':
                    # First add to knowledge base (ChromaDB)
                    kb_success = self.knowledge_base.add_example(description, scad_code, metadata=extracted_metadata)
                    
                    # Then log to conversation logs
                    self.logger.log_scad_generation(prompt_value, scad_code)
                    
                    if kb_success:
                        print("Example added to knowledge base and conversation logs!")
                        print("Your example is now immediately available for future generations.")
                    else:
                        print("Failed to add example to knowledge base, but it was saved to conversation logs.")
                else:
                    print("Example not saved. Thank you for the feedback!")
                
                # Save complete debug log
                self.save_debug_log()
        
                return {
                    'success': True,
                    'code': scad_code
                }
            
            except Exception as e:
                error_msg = f"Error generating OpenSCAD code: {str(e)}"
                print(f"\n{error_msg}")
                self.write_debug(
                    "=== GENERATION ERROR ===\n",
                    f"Error: {error_msg}\n",
                    "=" * 50 + "\n\n"
                )
                
                # Increment retry counter and continue if not max retries
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Retrying... ({retry_count}/{max_retries})")
                    continue
                else:
                    self.save_debug_log()
                    return {
                        'success': False,
                        'error': str(e)
                    }