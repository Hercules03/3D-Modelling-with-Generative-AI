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

    def perform_analysis(self, query: str, description: str, keyword_data, max_retries: int = 3, auto_approve: bool = True) -> Optional[Dict]:
        """Perform step-back analysis with user validation.
        
        Args:
            description: The description to analyze
            keyword_data: The extracted keyword data
            max_retries: Maximum number of retry attempts
            auto_approve: If True, automatically approve results (for testing)
            
        Returns:
            Dictionary containing step-back analysis if successful, None otherwise
        """
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Get the LLM model info
                print("\nUsing model for step-back analysis:")
                print(f"LLM type: {type(self.llm).__name__}")
                # Try to get model name from different attributes
                model_name = None
                if hasattr(self.llm, 'model'):
                    model_name = self.llm.model
                    print(f"Model name: {model_name}")
                elif hasattr(self.llm, 'model_name'):
                    model_name = self.llm.model_name
                    print(f"Model name: {model_name}")
                
                # Try to get base URL if available
                if hasattr(self.llm, 'base_url'):
                    print(f"Base URL: {self.llm.base_url}")
                print("-" * 50)
                
                # Format the step-back prompt with keyword data
                step_back_prompt_value = self.step_back_prompt.format(
                    Object=keyword_data.get('compound_type', ''),
                    Type=keyword_data.get('core_type', ''),
                    Modifiers=', '.join(keyword_data.get('modifiers', [])),
                    description=description
                )
                
                # Log analysis attempt
                self.write_debug(
                    "=== STEP-BACK ANALYSIS ===\n",
                    f"Attempt {retry_count + 1}/{max_retries}\n",
                    "Query:\n",
                    f"{query}\n\n",
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
                
                # Print the raw response for debugging
                print("\nRaw LLM Response:")
                print("-" * 50)
                print(technical_analysis)
                print("-" * 50)
                
                # Log raw response
                self.write_debug(
                    "Raw LLM Response:\n",
                    f"{technical_analysis}\n\n"
                )
                
                # Parse step-back analysis
                principles = []
                abstractions = []
                approach = []
                
                # More robust parsing - handle both exact and case-insensitive section headers
                current_section = None
                # First, extract the content inside <analysis> tags if present
                analysis_content = technical_analysis
                if "<analysis>" in technical_analysis.lower() and "</analysis>" in technical_analysis.lower():
                    start_idx = technical_analysis.lower().find("<analysis>") + len("<analysis>")
                    end_idx = technical_analysis.lower().find("</analysis>")
                    if start_idx < end_idx:
                        analysis_content = technical_analysis[start_idx:end_idx].strip()
                
                # Process the analysis content line by line
                for line in analysis_content.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Check for section headers with more flexibility
                    if any(header in line.upper() for header in ["CORE PRINCIPLES:", "CORE PRINCIPLES"]):
                        current_section = 'principles'
                        continue
                    elif any(header in line.upper() for header in ["SHAPE COMPONENTS:", "SHAPE COMPONENTS"]):
                        current_section = 'abstractions'
                        continue
                    elif any(header in line.upper() for header in ["IMPLEMENTATION STEPS:", "IMPLEMENTATION STEPS"]):
                        current_section = 'approach'
                        continue
                    elif "MEASUREMENT CONSIDERATIONS" in line.upper():
                        current_section = None  # Skip this section
                        continue
                    
                    # Process content based on current section
                    if current_section == 'principles':
                        # Handle bullet points with various markers and without markers
                        if line[0] in ['-', '•', '*'] or (line[0].isdigit() and '.' in line[:3]):
                            principles.append(line[line.find(' ')+1:].strip())
                        elif principles and line[0].isalpha():  # Continuation of previous point or unmarked point
                            principles.append(line)
                    elif current_section == 'abstractions':
                        if line[0] in ['-', '•', '*'] or (line[0].isdigit() and '.' in line[:3]):
                            abstractions.append(line[line.find(' ')+1:].strip())
                        elif abstractions and line[0].isalpha():  # Continuation or unmarked point
                            abstractions.append(line)
                    elif current_section == 'approach':
                        if line[0].isdigit() and '.' in line[:3]:
                            approach.append(line[line.find('.')+1:].strip())
                        elif line[0] in ['-', '•', '*']:
                            approach.append(line[line.find(' ')+1:].strip())
                        elif approach and line[0].isalpha():  # Continuation or unmarked point
                            approach.append(line)
                
                # If sections are still empty, try a fallback approach with simpler parsing
                if not any([principles, abstractions, approach]):
                    print("Warning: Initial parsing found no content, trying fallback parsing...")
                    # Try to identify paragraphs that might contain relevant information
                    paragraphs = [p.strip() for p in analysis_content.split('\n\n') if p.strip()]
                    for p in paragraphs:
                        if "principle" in p.lower() or "concept" in p.lower():
                            principles.append(p)
                        elif "shape" in p.lower() or "component" in p.lower() or "geometric" in p.lower():
                            abstractions.append(p)
                        elif "step" in p.lower() or "implement" in p.lower() or "approach" in p.lower():
                            approach.append(p)
                
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
                if auto_approve:
                    user_input = 'yes'
                    print("\nAuto-approving step-back analysis.")
                else:
                    user_input = input("\nDo you accept this technical analysis? (yes/no): ").lower().strip()
                
                # Log user's decision
                self.write_debug(
                    "=== USER STEP-BACK DECISION ===\n",
                    f"User accepted step-back analysis: {user_input == 'yes'} (Auto-approved: {auto_approve})\n",
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