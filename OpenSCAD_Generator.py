from prompts import OPENSCAD_GNERATOR_PROMPT_TEMPLATE, STEP_BACK_PROMPT_TEMPLATE, BASIC_KNOWLEDGE
from constant import *
from KeywordExtractor import KeywordExtractor
from LLM import LLMProvider
from SCADKnowledgeBase import KnowledgeBase
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
import datetime


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
            # "anthropic": "claude-3-5-sonnet-20240620",
            "anthropic": "claude-3-7-sonnet-20250219",
            "openai": "o1-mini",
            "gemma": "gemma3:4b-it-q8_0",
            "deepseek": "deepseek-r1:7b"
        }.get(llm_provider)
        
        self.llm = LLMProvider.get_llm(llm_provider)
        self.knowledge_base = KnowledgeBase()
        self.keyword_extractor = KeywordExtractor()
        
        # Create prompt templates
        self.main_prompt = ChatPromptTemplate.from_template(OPENSCAD_GNERATOR_PROMPT_TEMPLATE)
        self.step_back_prompt = ChatPromptTemplate.from_template(STEP_BACK_PROMPT_TEMPLATE)
    
    def perform_step_back(self, query):
        """Perform step-back prompting and get user validation"""
        while True:
            try:
                print("Generating step-back analysis...")
                
                # Write debug information before API call
                with open("debug.txt", "w") as f:
                    f.write("=== DEBUG INFORMATION ===\n\n")
                    f.write(f"Provider: {self.llm_provider}\n")
                    f.write(f"Model: {self.model_name}\n\n")
                
                try:
                    if self.llm_provider in ["anthropic", "openai"]:
                        message = HumanMessage(content=self.step_back_prompt.format(query=query))
                        response = self.llm.invoke([message])
                    else:
                        response = self.llm.invoke(self.step_back_prompt.format(query=query))
                    
                    # Write raw response to debug file
                    with open("debug.txt", "a") as f:
                        f.write("=== RAW RESPONSE ===\n")
                        f.write(str(response))
                        f.write("\n\n=== END RAW RESPONSE ===\n\n")
                    
                except Exception as e:
                    error_details = f"""
Error Details:
- Provider: {self.llm_provider}
- Model: {self.model_name}
- Error: {str(e)}
"""
                    with open("debug.txt", "a") as f:
                        f.write("\n=== ERROR DETAILS ===\n")
                        f.write(error_details)
                        f.write("\n=== END ERROR DETAILS ===\n")
                    raise Exception(f"API connection error with {self.llm_provider}: {str(e)}")
                
                # Extract content based on provider
                if hasattr(response, 'content'):
                    content = response.content
                else:
                    content = str(response)
                
                # Write parsed content to debug file
                with open("debug.txt", "a") as f:
                    f.write("=== PARSED CONTENT ===\n")
                    f.write(content)
                    f.write("\n=== END PARSED CONTENT ===\n")
                
                # Extract thinking process and analysis sections
                thinking_process = ""
                analysis_content = ""
                
                if '<think>' in content and '</think>' in content:
                    think_start = content.find('<think>') + len('<think>')
                    think_end = content.find('</think>')
                    thinking_process = content[think_start:think_end].strip()
                
                if '<analysis>' in content and '</analysis>' in content:
                    analysis_start = content.find('<analysis>') + len('<analysis>')
                    analysis_end = content.find('</analysis>')
                    analysis_content = content[analysis_start:analysis_end].strip()
                
                # Display the analysis section
                print("\nStep-back Analysis:")
                print("=" * 50)
                print(analysis_content)
                print("=" * 50)
                
                # Get user validation
                valid = input("\nIs this step-back analysis valid? (y/n): ").lower().strip()
                
                if valid == 'y':
                    # Parse the response into components
                    principles = []
                    abstractions = []
                    approach = []
                    
                    # Extract analysis components
                    if '<analysis>' in content and '</analysis>' in content:
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
                    principles_text = '\n'.join(f"- {p}" for p in principles)
                    abstractions_text = '\n'.join(f"- {a}" for a in abstractions)
                    approach_text = '\n'.join(f"{i+1}. {a}" for i, a in enumerate(approach))
                    
                    return {
                        'principles': principles_text,
                        'abstractions': abstractions_text,
                        'approach': approach_text
                    }
                else:
                    print("\nLet's try the step-back analysis again...")
                    
            except Exception as e:
                print(f"\nError in step-back analysis: {str(e)}")
                retry = input("Would you like to retry? (y/n): ").lower().strip()
                if retry != 'y':
                    return None
        
    def generate_model(self, description):
        """Generate OpenSCAD code for the given description and save it to a file."""
        try:
            # Step 1: Perform step-back prompting
            step_back_result = self.perform_step_back(description)
            if not step_back_result:
                print("Step-back analysis failed. Proceeding with basic generation...")
            
            # Step 2: Get relevant examples from knowledge base
            examples = self.knowledge_base.get_relevant_examples(description)
            
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
            
            # Write initial debug information
            with open("debug.txt", "w") as f:
                f.write("=== DEBUG INFORMATION ===\n\n")
                f.write(f"Provider: {self.llm_provider}\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n\n")
                f.write("=== FULL PROMPT ===\n")
                f.write(prompt_value)
                f.write("\n\n=== END PROMPT ===\n\n")
            
            if self.llm_provider in ["anthropic", "openai"]:
                try:
                    # Use HumanMessage format for ChatOpenAI
                    message = HumanMessage(content=prompt_value)
                    response = self.llm.invoke([message])
                except Exception as e:
                    error_details = f"""
                                    Error Details:
                                    - Provider: {self.llm_provider}
                                    - Model: {self.model_name}
                                    - Error: {str(e)}
                                    """
                    # Write error details to debug file
                    with open("debug.txt", "a") as f:
                        f.write("\n=== ERROR DETAILS ===\n")
                        f.write(error_details)
                        f.write("\n=== END ERROR DETAILS ===\n")
                    raise Exception(f"API connection error: {str(e)}")
            else:
                try:
                    # Use direct format for ChatOllama (gemma, deepseek)
                    response = self.llm.invoke(prompt_value)
                except Exception as e:
                    error_details = f"""
                                    Error Details:
                                    - Provider: {self.llm_provider}
                                    - Base URL: http://localhost:11434
                                    - Error: {str(e)}
                                    """
                    # Write error details to debug file
                    with open("debug.txt", "a") as f:
                        f.write("\n=== ERROR DETAILS ===\n")
                        f.write(error_details)
                        f.write("\n=== END ERROR DETAILS ===\n")
                    raise Exception(f"Ollama connection error: {str(e)}")
            
            # Extract content based on provider
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Initialize variables for thinking and code
            thinking_process = None
            scad_code = None
            
            # Extract thinking process and code using the unified format
            if '<think>' in content and '</think>' in content:
                think_start = content.find('<think>') + len('<think>')
                think_end = content.find('</think>')
                thinking_process = content[think_start:think_end].strip()
            
            if '<code>' in content and '</code>' in content:
                code_start = content.find('<code>') + len('<code>')
                code_end = content.find('</code>')
                scad_code = content[code_start:code_end].strip()
            
            # Write parsed components to debug file
            with open("debug.txt", "a") as f:
                f.write("=== PARSED COMPONENTS ===\n\n")
                f.write("--- Thinking Process ---\n")
                f.write(thinking_process if thinking_process else "No thinking process found")
                f.write("\n\n--- Code Extraction Process ---\n")
                if scad_code:
                    f.write("Code found within <code> tags\n")
                else:
                    f.write("Code not found within <code> tags, trying alternative formats...\n")
            
            # If code wasn't found in the expected format, try alternative formats
            if not scad_code:
                # Try markdown code blocks
                if '```' in content:
                    blocks = content.split('```')
                    for i in range(1, len(blocks), 2):
                        block = blocks[i].strip()
                        if block.startswith('scad'):
                            block = block[4:].strip()  # Remove 'scad' prefix
                        if block and not block.startswith('plaintext'):
                            scad_code = block
                            with open("debug.txt", "a") as f:
                                f.write("Code found in markdown block\n")
                            break
                
                # If still no code found, use the entire content
                if not scad_code:
                    scad_code = content.strip()
                    with open("debug.txt", "a") as f:
                        f.write("No code blocks found, using entire content\n")
            
            # Clean up the code
            if 'content="' in scad_code:
                content_start = scad_code.find('content="') + len('content="')
                content_end = scad_code.find('"', content_start)
                if content_end != -1:
                    scad_code = scad_code[content_start:content_end]
                    with open("debug.txt", "a") as f:
                        f.write("\nCleaned up code wrapped in content=\"...\"\n")
            
            scad_code = scad_code.replace('\\n', '\n')
            
            # Write final extracted code to debug file
            with open("debug.txt", "a") as f:
                f.write("\n=== FINAL EXTRACTED CODE ===\n")
                f.write(scad_code)
                f.write("\n\n=== END DEBUG INFORMATION ===\n")
            
            # Basic validation of the generated code
            if not scad_code or scad_code.strip() == '':
                raise ValueError("Generated code is empty or invalid")
            
            # Validate that the code matches the requested object
            keyword = self.keyword_extractor.extract_keyword(description)
            if keyword not in scad_code.lower() and keyword not in thinking_process.lower():
                print("\nWarning: Generated code may not match the requested object.")
                print("Regenerating with stronger emphasis on the requested object...")
                # Add explicit object requirement to the prompt
                inputs["request"] = f"{description}\n\nIMPORTANT: The code MUST be for a {keyword}. Do not generate code for any other object."
                prompt_value = self.main_prompt.format(**inputs)
                response = self.llm.invoke(prompt_value)
                # Extract and process the new response...
                if hasattr(response, 'content'):
                    content = response.content
                else:
                    content = str(response)
                
                # Re-extract the code with the same process
                if '<code>' in content and '</code>' in content:
                    code_start = content.find('<code>') + len('<code>')
                    code_end = content.find('</code>')
                    scad_code = content[code_start:code_end].strip()
            
            # Save to file
            with open("output.scad", "w") as f:
                f.write(scad_code)
            
            print("\nOpenSCAD code has been generated and saved to 'output.scad'")
            print("\nGenerated Code:")
            print("-" * 40)
            print(scad_code)
            print("-" * 40)
            
            """
            # Display thinking process if available
            if thinking_process:
                print("\nModel's Thinking Process:")
                print("-" * 40)
                print(thinking_process)
                print("-" * 40)
            """
            
            # Ask user if they want to add this to the knowledge base
            add_to_kb = input("\nWould you like to add this example to the knowledge base? (y/n): ").lower().strip()
            if add_to_kb == 'y':
                self.knowledge_base.add_example(description, scad_code)
                print("Example added to knowledge base!")
            
            return {
                'success': True,
                'code': scad_code,
                'thinking_process': thinking_process
            }
            
        except Exception as e:
            error_msg = f"Error generating model: {str(e)}"
            print(error_msg)
            print("\nDebug - Exception details:", e)
            # Write error to debug file
            with open("debug.txt", "a") as f:
                f.write(f"\nError occurred:\n{str(e)}\n")
            return {
                'success': False,
                'error': str(e),
                'code': None,
                'thinking_process': None
            }