from typing import Optional, Dict
from LLM import LLMProvider
from conversation_logger import ConversationLogger
from prompts import STEP_BACK_PROMPT_TEMPLATE
import datetime
import json
import logging

logger = logging.getLogger(__name__)

class StepBackAnalyzer:
    def __init__(self, llm, conversation_logger):
        """Initialize the step-back analyzer.
        
        Args:
            llm: Optional LLM provider instance. If not provided, will create a new one.
            logger: Optional conversation logger instance. If not provided, will create a new one.
        """
        print("Step-back analyzer initialising...")
        logger.info("Step-back analyzer initialising...")
        self.llm = llm
        self.conversation_logger = conversation_logger
        self.step_back_prompt = STEP_BACK_PROMPT_TEMPLATE
        self.debug_log = []
        logger.info("Step-back analyzer initialised")
        print("Step-back analyzer initialised")

    def write_debug(self, *messages):
        """Write messages to debug log"""
        for message in messages:
            self.debug_log.append(message)

    def perform_analysis(self, description: str, keyword_data: Dict, max_retries: int = 3) -> Optional[Dict]:
        """Perform step-back analysis with user validation.
        
        Args:
            description: The description to analyze
            keyword_data: The extracted keyword data
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary containing step-back analysis if successful, None otherwise
        """
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Format the step-back prompt with keyword data
                step_back_prompt_value = self.step_back_prompt.format(
                    Object=keyword_data.get('compound_type', ''),
                    Type=keyword_data.get('core_type', ''),
                    Modifiers=', '.join(keyword_data.get('modifiers', []))
                )
                
                # Log analysis attempt
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
                
                # Get response from LLM
                response = self.llm.invoke(step_back_prompt_value)
                technical_analysis = response.content
                
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
                    elif line and line[0] in ['-', '•', '*'] and current_section == 'principles':
                        principles.append(line[1:].strip())
                    elif line and line[0] in ['-', '•', '*'] and current_section == 'abstractions':
                        abstractions.append(line[1:].strip())
                    elif line and line[0].isdigit() and current_section == 'approach':
                        approach.append(line[line.find('.')+1:].strip())
                
                # Create analysis result
                analysis_result = {
                    'principles': principles,
                    'abstractions': abstractions,
                    'approach': approach,
                    'original_query': description,
                    'keyword_analysis': keyword_data
                }
                
                # Log analysis results
                self.write_debug(
                    "Response:\n",
                    "Core Principles:\n",
                    "\n".join(f"- {p}" for p in principles) + "\n\n",
                    "Shape Components:\n",
                    "\n".join(f"- {a}" for a in abstractions) + "\n\n",
                    "Implementation Steps:\n",
                    "\n".join(f"{i+1}. {s}" for i, s in enumerate(approach)) + "\n",
                    "=" * 50 + "\n\n"
                )
                
                # Display results to user
                print("\nStep-back Analysis Results:")
                print("-" * 30)
                print("\nCore Principles:")
                for p in principles:
                    print(f"- {p}")
                print("\nShape Components:")
                for a in abstractions:
                    print(f"- {a}")
                print("\nImplementation Steps:")
                for i, s in enumerate(approach, 1):
                    print(f"{i}. {s}")
                print("-" * 30)
                
                # Get user confirmation
                user_input = input("\nDo you accept this technical analysis? (yes/no): ").lower().strip()
                
                # Log user's decision
                self.write_debug(
                    "=== USER STEP-BACK DECISION ===\n",
                    f"User accepted step-back analysis: {user_input == 'yes'}\n",
                    "=" * 50 + "\n\n"
                )
                
                if user_input == 'yes':
                    # Log the approved analysis
                    self.conversation_logger.log_step_back_analysis({
                        "query": {
                            "input": description,
                            "timestamp": datetime.datetime.now().isoformat()
                        },
                        "response": {
                            "principles": principles,
                            "abstractions": abstractions,
                            "approach": approach
                        },
                        "metadata": {
                            "success": True,
                            "error": None,
                            "user_approved": True
                        }
                    })
                    return analysis_result
                
                retry_count += 1
                if retry_count < max_retries:
                    print("\nRetrying step-back analysis...")
                    # Ask user for refinement suggestions
                    print("Please provide any suggestions to improve the analysis (or press Enter to retry):")
                    user_feedback = input().strip()
                    if user_feedback:
                        description = f"{description}\nConsider these aspects in your analysis: {user_feedback}"
                else:
                    print("\nMaximum step-back analysis attempts reached.")
                    print("Please try again with a different description.")
                    
            except Exception as e:
                error_msg = f"Error in step-back analysis: {str(e)}"
                print(f"\n{error_msg}")
                self.write_debug(
                    "=== STEP-BACK ERROR ===\n",
                    f"{error_msg}\n",
                    "=" * 50 + "\n\n"
                )
                retry_count += 1
                if retry_count < max_retries:
                    print(f"\nRetrying... ({retry_count}/{max_retries})")
                else:
                    return None
        
        return None 