from prompts import OPENSCAD_GNERATOR_PROMPT_TEMPLATE, STEP_BACK_PROMPT_TEMPLATE, BASIC_KNOWLEDGE
from constant import *
from KeywordExtractor import KeywordExtractor
from llm_management import LLMProvider
from enhanced_scad_knowledge_base import EnhancedSCADKnowledgeBase
from langchain.prompts import ChatPromptTemplate
import datetime
from conversation_logger import ConversationLogger
from typing import Optional


class OpenSCADGenerator:
    def __init__(self, llm_provider="anthropic"):
        """Initialize the OpenSCAD generator"""
        print("\n=== Initializing OpenSCAD Generator ===")
        
        # Initialize LLM
        print("\nSetting up LLM...")
        self.llm_provider = llm_provider
        self.llm = LLMProvider.get_llm(provider=llm_provider)
        self.model_name = self.llm.model_name if hasattr(self.llm, 'model_name') else str(self.llm)
        print(f"- Provider: {self.llm_provider}")
        print(f"- Model: {self.model_name}")
        
        # Initialize knowledge base and logger
        print("\nSetting up components...")
        self.knowledge_base = EnhancedSCADKnowledgeBase()
        self.logger = ConversationLogger()
        print("- Knowledge base initialized")
        print("- Conversation logger initialized")
        
        # Load prompts
        print("\nLoading prompts...")
        self.step_back_prompt = STEP_BACK_PROMPT_TEMPLATE
        self.main_prompt = OPENSCAD_GNERATOR_PROMPT_TEMPLATE
        print("- Step-back prompt loaded")
        print("- Main generation prompt loaded")
        
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
    
    def perform_step_back(self, query):
        """Perform step-back prompting and get user validation"""
        while True:
            try:
                # Get step-back analysis with focused description from knowledge base
                analysis_result = self.knowledge_base.perform_step_back(query)
                if not analysis_result:
                    msg = "Step-back analysis failed. Proceeding with basic generation..."
                    print(msg)
                    self.write_debug(f"{msg}\n\n")
                    return None

                # Extract the focused description and analysis components
                focused_description = analysis_result.get('focused_description', query)
                original_query = analysis_result.get('original_query', query)
                
                print("\nStep-back Analysis Results:")
                print(f"Original Query: {original_query}")
                print(f"Focused Description: {focused_description}")
                print("\nCore Principles:")
                for principle in analysis_result.get('principles', []):
                    print(f"- {principle}")
                print("\nShape Components:")
                for component in analysis_result.get('abstractions', []):
                    print(f"- {component}")
                print("\nImplementation Steps:")
                for step in analysis_result.get('approach', []):
                    print(f"- {step}")

                # Get user validation
                print("\nDoes this analysis look correct? (y/n): ")
                valid = input().lower().strip()
                
                if valid == 'y':
                    # Only log step-back conversation if user accepts it
                    self.logger.log_step_back(focused_description, analysis_result)
                    return analysis_result
                else:
                    print("\nLet's try the step-back analysis again...")
                    self.write_debug("Retrying step-back analysis...\n\n")
                    
            except Exception as e:
                error_msg = f"Error in step-back analysis: {str(e)}"
                print(f"\n{error_msg}")
                # Add error to debug log
                self.write_debug(
                    "\n=== STEP-BACK ERROR ===\n",
                    f"{error_msg}\n",
                    "=" * 50 + "\n\n"
                )
                retry = input("Would you like to retry? (y/n): ").lower().strip()
                if retry != 'y':
                    return None
        
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
        
        while retry_count < max_retries:
            # Log keyword extraction query and prompt
            keyword_prompt = self.knowledge_base.keyword_extractor.prompt.replace("<<INPUT>>", description)
            self.write_debug(
                "=== KEYWORD EXTRACTION ===\n",
                f"Attempt {retry_count + 1}/{max_retries}\n",
                "Query:\n",
                f"{description}\n\n",
                "Full Prompt Sent to LLM:\n",
                f"{keyword_prompt}\n\n"
            )
            
            keyword_data = self.knowledge_base.keyword_extractor.extract_keyword(description)
            
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

    def perform_step_back_analysis(self, description: str, keyword_data: dict, max_retries: int = 3) -> Optional[dict]:
        """Perform step-back analysis with approved keywords.
        
        Args:
            description: The description to analyze
            keyword_data: The extracted keyword data
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary containing step-back analysis if successful, None otherwise
        """
        retry_count = 0
        
        while retry_count < max_retries:
            # Log step-back analysis query and prompt
            step_back_prompt_value = self.step_back_prompt.format(
                Object=keyword_data.get('compound_type', '') or keyword_data.get('core_type', ''),
                Type=keyword_data.get('core_type', ''),
                Modifiers=', '.join(keyword_data.get('modifiers', []))
            )
            
            self.write_debug(
                "=== STEP-BACK ANALYSIS ===\n",
                f"Attempt {retry_count + 1}/{max_retries}\n",
                "Query:\n",
                f"{description}\n\n",
                "Keyword Data:\n",
                f"Core Type: {keyword_data.get('core_type', '')}\n",
                f"Modifiers: {', '.join(keyword_data.get('modifiers', []))}\n",
                f"Compound Type: {keyword_data.get('compound_type', '')}\n\n",
                "Full Prompt Sent to LLM:\n",
                f"{step_back_prompt_value}\n\n"
            )
            
            step_back_result = self.knowledge_base.perform_step_back(description, keyword_data)
            if not step_back_result:
                msg = "Step-back analysis failed. Proceeding with basic generation..."
                print(msg)
                self.write_debug(f"{msg}\n\n")
                return None
            
            # Log step-back analysis results
            self.write_debug(
                "Response:\n",
                "Core Principles:\n",
                "\n".join(f"- {p}" for p in step_back_result.get('principles', [])) + "\n\n",
                "Shape Components:\n",
                "\n".join(f"- {a}" for a in step_back_result.get('abstractions', [])) + "\n\n",
                "Implementation Steps:\n",
                "\n".join(f"{i+1}. {s}" for i, s in enumerate(step_back_result.get('approach', []))) + "\n",
                "=" * 50 + "\n\n"
            )
            
            # Ask for user confirmation
            user_input = input("\nDo you accept this technical analysis? (yes/no): ").lower().strip()
            
            # Log user's step-back decision
            self.write_debug(
                "=== USER STEP-BACK DECISION ===\n",
                f"User accepted step-back analysis: {user_input == 'yes'}\n",
                "=" * 50 + "\n\n"
            )
            
            if user_input == 'yes':
                # Log the approved step-back analysis
                self.logger.log_step_back_analysis({
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
                return step_back_result
            
            retry_count += 1
            if retry_count < max_retries:
                print("\nRetrying step-back analysis...")
                # Ask user for refinement suggestions
                print("Please provide any suggestions to improve the step-back analysis (or press Enter to retry):")
                user_feedback = input().strip()
                if user_feedback:
                    description = f"{description}\nConsider these aspects in your analysis: {user_feedback}"
            else:
                print("\nMaximum step-back analysis attempts reached.")
                print("Please try again with a different description.")
        
        return None

    def retrieve_similar_examples(self, description: str, step_back_result: dict, keyword_data: dict) -> tuple[list, Optional[dict]]:
        """Retrieve similar examples from the knowledge base.
        
        Args:
            description: The description to find examples for
            step_back_result: The step-back analysis results
            keyword_data: The extracted keyword data
            
        Returns:
            Tuple of (examples list, extracted metadata)
        """
        print("\nRetrieving relevant examples...")
        
        # Log retrieval query with full search parameters
        self.write_debug(
            "=== EXAMPLE RETRIEVAL ===\n",
            "Query:\n",
            f"Description: {description}\n",
            "Filters from Step-back Analysis:\n",
            f"- Principles: {', '.join(step_back_result.get('principles', []))}\n",
            f"- Components: {', '.join(step_back_result.get('abstractions', []))}\n\n",
            "Keyword Data for Filtering:\n",
            f"Core Type: {keyword_data.get('core_type', '')}\n",
            f"Modifiers: {', '.join(keyword_data.get('modifiers', []))}\n",
            f"Compound Type: {keyword_data.get('compound_type', '')}\n\n",
            "Search Parameters:\n",
            f"Similarity Threshold: {0.7}\n\n"
        )
        
        examples, extracted_metadata = self.knowledge_base.get_similar_examples(
            description,
            step_back_result=step_back_result,
            keyword_data=keyword_data,
            return_metadata=True
        )
        
        # Log retrieval results
        self.write_debug(
            "Retrieved Examples:\n",
            f"Total Examples Found: {len(examples)}\n\n"
        )
        
        return examples, extracted_metadata

    def log_ranked_examples(self, examples: list):
        """Log the ranked examples and their scores.
        
        Args:
            examples: List of ranked examples
        """
        if examples:
            for i, ex in enumerate(examples, 1):
                example_id = ex.get('example', {}).get('id', 'unknown')
                score = ex.get('score', 0.0)
                score_breakdown = ex.get('score_breakdown', {})
                
                self.write_debug(
                    f"Example {i}:\n",
                    f"ID: {example_id}\n",
                    f"Score: {score:.3f}\n",
                    "Component Scores:\n",
                    "\n".join(f"  - {name}: {score:.3f}" 
                            for name, score in score_breakdown.get('component_scores', {}).items()),
                    "\n\n"
                )
        
        # Log re-ranking results
        self.write_debug(
            "=== RE-RANKING RESULTS ===\n",
            f"Number of examples after re-ranking: {len(examples)}\n\n",
            "Final Ranked Examples:\n"
        )
        
        for i, ex in enumerate(examples, 1):
            self.write_debug(
                f"Rank {i}:\n",
                f"ID: {ex.get('example', {}).get('id', 'unknown')}\n",
                f"Final Score: {ex.get('score', 0.0):.3f}\n",
                "Score Components:\n",
                "\n".join(f"  - {name}: {score:.3f}"
                        for name, score in ex.get('score_breakdown', {}).get('component_scores', {}).items()),
                "\n\n"
            )
        
        self.write_debug("=" * 50 + "\n\n")

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
        inputs = self.knowledge_base.prepare_generation_inputs(
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
                    self.log_ranked_examples(examples)
                
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
                    # Only log successful and approved generations
                    self.logger.log_scad_generation(prompt_value, scad_code)
                    # Add to knowledge base if user approves, using previously extracted metadata
                    if self.knowledge_base.add_example(description, scad_code, metadata=extracted_metadata):
                        print("Example added to knowledge base!")
                    else:
                        print("Failed to add example to knowledge base.")
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

    def view_knowledge_base(self):
        """View and explore the contents of the knowledge base.
        
        This method allows users to:
        1. View all examples in the knowledge base
        2. Select a specific example by index
        3. View the full details of the selected example including SCAD code
        """
        try:
            # Get all examples from the knowledge base
            print("\n" + "="*50)
            print("KNOWLEDGE BASE EXPLORER")
            print("="*50)
            
            examples = self.knowledge_base.get_all_examples()
            if not examples:
                print("\nNo examples found in the knowledge base.")
                return
            
            # Display examples with their descriptions and metadata
            print(f"\nFound {len(examples)} examples:")
            print("-" * 40)
            
            for i, example in enumerate(examples, 1):
                metadata = example.get('metadata', {})
                description = example.get('description', 'No description available')
                object_type = metadata.get('object_type', 'Unknown type')
                features = ', '.join(metadata.get('features', [])) or 'No features'
                timestamp = metadata.get('timestamp', 'Unknown time')
                
                print(f"\n[{i}] Example ID: {example.get('id', 'unknown')}")
                print(f"Description: {description}")
                print(f"Type: {object_type}")
                print(f"Features: {features}")
                print(f"Added: {timestamp}")
                print("-" * 40)
            
            while True:
                try:
                    # Get user selection
                    selection = input("\nEnter the number of the example to view (or 'q' to quit): ").lower().strip()
                    
                    if selection == 'q':
                        return
                    
                    index = int(selection) - 1
                    if 0 <= index < len(examples):
                        selected = examples[index]
                        
                        # Display full example details
                        print("\n" + "="*50)
                        print("EXAMPLE DETAILS")
                        print("="*50)
                        
                        print("\nDescription:")
                        print(selected.get('description', 'No description available'))
                        
                        print("\nMetadata:")
                        metadata = selected.get('metadata', {})
                        print(f"- Object Type: {metadata.get('object_type', 'Unknown')}")
                        print(f"- Features: {', '.join(metadata.get('features', []))}")
                        print(f"- Added: {metadata.get('timestamp', 'Unknown')}")
                        print(f"- Type: {metadata.get('type', 'Unknown')}")
                        
                        if 'step_back_analysis' in metadata:
                            analysis = metadata['step_back_analysis']
                            print("\nTechnical Analysis:")
                            if 'principles' in analysis:
                                print("\nCore Principles:")
                                for p in analysis['principles']:
                                    print(f"- {p}")
                            if 'abstractions' in analysis:
                                print("\nShape Components:")
                                for a in analysis['abstractions']:
                                    print(f"- {a}")
                            if 'approach' in analysis:
                                print("\nImplementation Steps:")
                                for i, s in enumerate(analysis['approach'], 1):
                                    print(f"{i}. {s}")
                        
                        print("\nOpenSCAD Code:")
                        print("-" * 40)
                        print(selected.get('code', 'No code available'))
                        print("-" * 40)
                        
                        # Ask if user wants to save this code to a file
                        save = input("\nWould you like to save this code to a file? (y/n): ").lower().strip()
                        if save == 'y':
                            filename = f"example_{selected.get('id', 'unknown')}.scad"
                            with open(filename, "w") as f:
                                f.write(selected.get('code', ''))
                            print(f"\nCode saved to {filename}")
                        
                        # Ask if user wants to view another example
                        again = input("\nWould you like to view another example? (y/n): ").lower().strip()
                        if again != 'y':
                            break
                    else:
                        print("\nInvalid selection. Please enter a number between 1 and", len(examples))
                except ValueError:
                    print("\nInvalid input. Please enter a number or 'q' to quit.")
                except Exception as e:
                    print(f"\nError viewing example: {str(e)}")
        
        except Exception as e:
            print(f"\nError accessing knowledge base: {str(e)}")
            self.write_debug(
                "=== KNOWLEDGE BASE VIEW ERROR ===\n",
                f"Error: {str(e)}\n",
                "=" * 50 + "\n\n"
            )