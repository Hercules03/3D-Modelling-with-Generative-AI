from prompts import OPENSCAD_GNERATOR_PROMPT_TEMPLATE, STEP_BACK_PROMPT_TEMPLATE, BASIC_KNOWLEDGE
from constant import *
from KeywordExtractor import KeywordExtractor
from LLM import LLMProvider
from SCADKnowledgeBase import KnowledgeBase
from langchain.prompts import ChatPromptTemplate
import datetime
from conversation_logger import ConversationLogger
from LLMmodel import *


class OpenSCADGenerator:
    def __init__(self, llm_provider="anthropic"):
        """
        Initialize the OpenSCAD generator
        Args:
            llm_provider (str): LLM provider to use ('anthropic', 'openai', 'gemma', 'deepseek')
        """
        self.llm_provider = llm_provider
        # Store model name based on provider
        self.model_name = {
            "anthropic": anthropic_model,
            "openai": openai_model,
            "gemma": gemma_model,
            "deepseek": deepseek_model
        }.get(llm_provider)
        
        self.llm = LLMProvider.get_llm(llm_provider)
        self.knowledge_base = KnowledgeBase()
        self.keyword_extractor = KeywordExtractor()
        self.logger = ConversationLogger()
        
        # Create prompt templates
        self.main_prompt = ChatPromptTemplate.from_template(OPENSCAD_GNERATOR_PROMPT_TEMPLATE)
        self.step_back_prompt = ChatPromptTemplate.from_template(STEP_BACK_PROMPT_TEMPLATE)
        
        # Initialize debug log
        self.debug_log = []
    
    def write_debug(self, *messages):
        """Add messages to debug log"""
        self.debug_log.extend(messages)
        
    def save_debug_log(self):
        """Write the complete debug log to debug.txt"""
        with open("debug.txt", "w") as f:
            f.writelines(self.debug_log)
    
    def perform_step_back(self, query):
        """Perform step-back prompting and get user validation"""
        while True:
            try:
                # Format the prompt
                prompt_value = self.step_back_prompt.format(query=query)
                print("Performing step-back analysis...")
                
                # Add to debug log
                self.write_debug(
                    "\n=== STEP-BACK ANALYSIS ===\n",
                    f"Query: {query}\n",
                    f"Provider: {self.llm_provider}\n",
                    f"Model: {self.model_name}\n\n",
                    "=== FULL PROMPT ===\n",
                    f"{prompt_value}\n\n"
                )
                
                # Get response from LLM
                response = self.llm.invoke(prompt_value)
                
                # Extract content
                if hasattr(response, 'content'):
                    content = response.content
                else:
                    content = str(response)
                
                # Add response to debug log
                self.write_debug(
                    "=== LLM RESPONSE ===\n",
                    f"{content}\n\n"
                )
                
                # Extract analysis components
                analysis_start = content.find('<analysis>') + len('<analysis>')
                analysis_end = content.find('</analysis>')
                
                if '<analysis>' not in content or '</analysis>' not in content:
                    error_msg = "Could not find analysis section in response"
                    print(f"\nError: {error_msg}")
                    # Add error to debug log
                    self.write_debug(
                        "=== ANALYSIS ERROR ===\n",
                        f"Error: {error_msg}\n",
                        f"Raw Response:\n{content}\n",
                        "=" * 50 + "\n\n"
                    )
                    return None
                
                # Parse the response into components
                principles = []
                abstractions = []
                approach = []
                
                # Extract analysis components
                analysis = content[analysis_start:analysis_end].strip()
                sections = analysis.split('\n')
                current_section = None
                
                for line in sections:
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
                
                # Format the components
                analysis_result = {
                    'principles': '\n'.join(f"- {p}" for p in principles),
                    'abstractions': '\n'.join(f"- {a}" for a in abstractions),
                    'approach': '\n'.join(f"{i+1}. {a}" for i, a in enumerate(approach))
                }
                
                # Add parsed analysis to debug log
                self.write_debug(
                    "=== PARSED COMPONENTS ===\n",
                    "\nCORE PRINCIPLES:\n",
                    f"{analysis_result['principles']}\n",
                    "\nSHAPE COMPONENTS:\n",
                    f"{analysis_result['abstractions']}\n",
                    "\nIMPLEMENTATION STEPS:\n",
                    f"{analysis_result['approach']}\n\n"
                )
                
                # Display the analysis section
                print("\nStep-back Analysis:")
                print("=" * 50)
                print("CORE PRINCIPLES:")
                print(analysis_result['principles'])
                print("\nSHAPE COMPONENTS:")
                print(analysis_result['abstractions'])
                print("\nIMPLEMENTATION STEPS:")
                print(analysis_result['approach'])
                print("=" * 50)
                
                # Get user validation
                valid = input("\nIs this step-back analysis valid? (y/n): ").lower().strip()
                
                # Add validation result to debug log
                self.write_debug(
                    "=== USER VALIDATION ===\n",
                    f"Valid: {valid}\n\n"
                )
                
                if valid == 'y':
                    # Only log step-back conversation if user accepts it
                    self.logger.log_step_back(query, analysis_result)
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
    
    def generate_model(self, description):
        """Generate OpenSCAD code for the given description and save it to a file."""
        try:
            # Clear previous debug log
            self.debug_log = []
            
            # Add generation start to debug log
            self.write_debug(
                "=== SCAD GENERATION START ===\n",
                f"Description: {description}\n",
                f"Provider: {self.llm_provider}\n",
                f"Model: {self.model_name}\n",
                "=" * 50 + "\n\n"
            )
            
            # Step 1: Perform step-back prompting
            step_back_result = self.perform_step_back(description)
            if not step_back_result:
                msg = "Step-back analysis failed. Proceeding with basic generation..."
                print(msg)
                self.write_debug(f"{msg}\n\n")
            
            # Step 2: Get relevant examples from knowledge base
            examples = self.knowledge_base.get_relevant_examples(description)
            
            # Add examples to debug log
            self.write_debug(
                "=== RETRIEVED EXAMPLES ===\n",
                f"{examples}\n",
                "=" * 50 + "\n\n"
            )
            
            # Step 3: Prepare the prompt inputs
            inputs = {
                "basic_knowledge": BASIC_KNOWLEDGE,
                "examples": examples,
                "request": description,
                "step_back_analysis": ""
            }
            
            # Add step-back analysis if available
            if step_back_result:
                step_back_text = f"""
                CORE PRINCIPLES:
                {step_back_result['principles']}
                
                SHAPE COMPONENTS:
                {step_back_result['abstractions']}
                
                IMPLEMENTATION STEPS:
                {step_back_result['approach']}
                """
                inputs["step_back_analysis"] = step_back_text.strip()
            
            # Generate the prompt and get response
            print("Generating OpenSCAD code...")
            prompt_value = self.main_prompt.format(**inputs)
            
            # Add generation prompt to debug log
            self.write_debug(
                "=== GENERATION PROMPT ===\n",
                f"{prompt_value}\n\n"
            )
            
            response = self.llm.invoke(prompt_value)
            
            # Extract content
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Add LLM response to debug log
            self.write_debug(
                "=== LLM RESPONSE ===\n",
                f"{content}\n\n"
            )
            
            # Extract code and thinking process
            if '<code>' in content and '</code>' in content:
                code_start = content.find('<code>') + len('<code>')
                code_end = content.find('</code>')
                scad_code = content[code_start:code_end].strip()
                
                # Save to file
                with open("output.scad", "w") as f:
                    f.write(scad_code)
                
                # Add generated code to debug log
                self.write_debug(
                    "=== GENERATED SCAD CODE ===\n",
                    f"{scad_code}\n",
                    "=" * 50 + "\n\n"
                )
                
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
                    self.knowledge_base.add_example(description, scad_code)
                    print("Example added to knowledge base and logged for future reference!")
                else:
                    print("Example not saved. Thank you for the feedback!")
                
                # Save complete debug log
                self.save_debug_log()
                
                return {
                    'success': True,
                    'code': scad_code
                }
            else:
                error_msg = "No code section found in response"
                self.write_debug(
                    "=== GENERATION ERROR ===\n",
                    f"Error: {error_msg}\n",
                    "=" * 50 + "\n\n"
                )
                self.save_debug_log()
                raise ValueError(error_msg)
                
        except Exception as e:
            error_msg = f"Error generating OpenSCAD code: {str(e)}"
            print(f"\n{error_msg}")
            self.write_debug(
                "=== GENERATION ERROR ===\n",
                f"Error: {error_msg}\n",
                "=" * 50 + "\n\n"
            )
            self.save_debug_log()
            return {
                'success': False,
                'error': str(e)
            }