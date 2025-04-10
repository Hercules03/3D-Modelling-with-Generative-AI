from prompts import OPENSCAD_GNERATOR_PROMPT_TEMPLATE, BASIC_KNOWLEDGE
from constant import *
from LLM import LLMProvider
from step_back_analyzer import StepBackAnalyzer
from parameter_tuner import ParameterTuner, identify_parameters_from_examples, get_user_parameter_input
import datetime
from typing import Optional, List, Dict, Literal, Union
import logging
import traceback
import os

from session_integration import SessionIntegration
from model_exporter import ModelExporter
from models import GenerationResult, StepBackAnalysis, KeywordData, ExportResult

logger = logging.getLogger(__name__)

class OpenSCADGenerator:
    def __init__(self, llm_provider="anthropic", knowledge_base=None, keyword_extractor=None, metadata_extractor=None, conversation_logger=None, session_integration=None, output_dir="output"):
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
        
        # Initialize session integration
        self.session_integration = session_integration
        
        # Initialize model exporter
        self.output_dir = output_dir
        self.model_exporter = ModelExporter(output_dir=output_dir)
        
        # Initialize step-back analyzer
        print("- Initializing step-back analyzer...")
        logger.info("Initializing step-back analyzer...")
        self.step_back_analyzer = StepBackAnalyzer(llm=self.llm, conversation_logger=self.logger)
        print("- Step-back analyzer initialized")
        logger.info("Step-back analyzer initialized")
        
        # Initialize parameter tuner
        self.parameter_tuner = ParameterTuner(llm=self.llm)
        print("- Parameter tuner initialized")
        logger.info("- Parameter tuner initialized")
        
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

    def prepare_generation_inputs(self, description: str, examples: List[Dict], step_back_result: Dict = None, parameters: Dict = None) -> Dict:
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
            # Import template utilities
            from scad_templates import select_template_for_object, generate_template_params, apply_template
            
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
            
            # Get object type and modifiers from keyword extraction
            object_type = ""
            modifiers = []
            if hasattr(self, 'keyword_extractor') and hasattr(self.keyword_extractor, 'last_extracted_data'):
                object_type = self.keyword_extractor.last_extracted_data.get('core_type', '')
                modifiers = self.keyword_extractor.last_extracted_data.get('modifiers', [])
            
            # Select appropriate template based on object characteristics
            template_name = select_template_for_object(object_type, modifiers, step_back_result)
            
            # Generate template parameters
            template_params = generate_template_params(object_type, modifiers, step_back_result)
            
            # Apply the template to get example code
            template_code = apply_template(template_name, template_params)
            
            # Add template information to the inputs
            template_info = f"""
            SUGGESTED TEMPLATE:
            The object appears to be a "{template_name}" type. Here's a suggested structure:
            
            ```scad
            {template_code}
            ```
            
            Feel free to use this template as a starting point and modify it as needed.
            """
            
            # Add parameter information if provided
            parameter_info = ""
            if parameters:
                parameter_info = "SUGGESTED PARAMETERS:\n"
                for name, info in parameters.items():
                    value = info.get("value")
                    description = info.get("description", "")
                    # Format based on type
                    if isinstance(value, list):
                        # Vector format
                        value_str = f"[{', '.join(str(x) for x in value)}]"
                    elif isinstance(value, bool):
                        # Boolean format
                        value_str = "true" if value else "false"
                    else:
                        value_str = str(value)
                    
                    parameter_info += f"{name} = {value_str}; // {description}\n"
            
            # Prepare the complete inputs
            inputs = {
                "basic_knowledge": BASIC_KNOWLEDGE,
                "examples": examples,
                "request": description,
                "step_back_analysis": step_back_text.strip() if step_back_text else "",
                "template_suggestion": template_info,
                "parameter_suggestions": parameter_info
            }
            
            # Log the complete analysis and examples
            print("\nSelected Template: " + template_name)
            
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
                "step_back_analysis": "",
                "template_suggestion": ""
            }

    def generate_scad_code(self, description: str, examples: list, step_back_result: dict, parameters: dict = None) -> Optional[dict]:
        """Generate OpenSCAD code using the prepared inputs.
        
        Args:
            description: The description to generate code for
            examples: List of similar examples
            step_back_result: The step-back analysis results
            parameters: Dictionary of parameters to use
            
        Returns:
            Dictionary containing success status and generated code/error
        """
        # Import the graph-based code generation function
        from graph_state_tools import create_generate_scad_code_function
        
        # First prepare all the inputs as we normally would
        inputs = self.prepare_generation_inputs(
            description=description,
            examples=examples,
            step_back_result=step_back_result,
            parameters=parameters
        )
        
        # Create a minimal state object with all needed info
        state = {
            "input_text": description,
            "step_back_result": step_back_result or {},
            "parameters": parameters or {},
            "examples": examples or [],
            "prompt_inputs": inputs
        }
        
        print("Generating OpenSCAD code using graph-based system...")
        print("Thinking...", end="", flush=True)
        
        try:
            # Create the code generation function from graph_state_tools
            generate_code_fn = create_generate_scad_code_function(self.llm, None)
            
            # Call the code generation function with our state
            result = generate_code_fn(state)
            
            print("\n")  # New line after progress dots
            
            # Extract the result and format it as expected by the rest of the system
            if "generated_code" in result:
                if result["generated_code"].get("success", False):
                    success_result = GenerationResult(
                        success=True,
                        code=result["generated_code"]["code"],
                        prompt=inputs.get("prompt_value", "")
                    )
                    return success_result.model_dump()
                else:
                    error_result = GenerationResult(
                        success=False,
                        error=result["generated_code"].get("error", "Generation failed")
                    )
                    return error_result.model_dump()
            elif isinstance(result, dict) and "code" in result:
                # Handle simpler result format
                success_result = GenerationResult(
                    success=True,
                    code=result["code"],
                    prompt=inputs.get("prompt_value", "")
                )
                return success_result.model_dump()
            else:
                # Fallback to error
                error_result = GenerationResult(
                    success=False,
                    error="Graph-based code generation failed with unexpected result format"
                )
                return error_result.model_dump()
                
        except Exception as e:
            error_msg = f"Error in graph-based code generation: {str(e)}"
            print(f"\n{error_msg}")
            error_result = GenerationResult(
                success=False,
                error=error_msg
            )
            return error_result.model_dump()

    def export_model(self, scad_code, format="stl", filename="model", resolution=100, additional_params=None):
        """Export SCAD code to different formats using ModelExporter
        
        Args:
            scad_code: OpenSCAD code to export
            format: Format to export to (stl, off, amf, 3mf, csg, dxf, svg, png)
            filename: Name for the output file (without extension)
            resolution: Resolution for curves ($fn value)
            additional_params: Additional parameters to pass to OpenSCAD
            
        Returns:
            ExportResult object with export details
        """
        logger.info(f"Exporting model to {format} format with resolution {resolution}")
        
        try:
            # Call the model exporter
            output_path = self.model_exporter.export_model(
                scad_code=scad_code,
                filename=filename,
                export_format=format,
                resolution=resolution,
                additional_params=additional_params
            )
            
            if output_path:
                # Create successful result
                result = ExportResult(
                    success=True,
                    format=format,
                    file_path=output_path,
                    file_size=os.path.getsize(output_path) if os.path.exists(output_path) else None
                )
                logger.info(f"Successfully exported model to {output_path}")
                return result
            else:
                # Create error result
                result = ExportResult(
                    success=False,
                    format=format,
                    error="Export failed"
                )
                logger.error(f"Failed to export model to {format} format")
                return result
                
        except Exception as e:
            error_msg = f"Error exporting model: {str(e)}"
            logger.error(error_msg)
            return ExportResult(
                success=False,
                format=format,
                error=error_msg
            )
    
    def generate_model_preview(self, scad_code: str, filename: str = "preview", 
                         resolution: int = 100, camera_params: str = "0,0,0,55,0,25,140", 
                         width: int = 800, height: int = 600) -> Optional[str]:
        """Generate a preview image of the model
        
        Args:
            scad_code (str): The OpenSCAD code to render
            filename (str): Base name for the output file (without extension)
            resolution (int): Resolution for curves ($fn value)
            camera_params (str): Camera position for the render
            width (int): Image width in pixels
            height (int): Image height in pixels
            
        Returns:
            Optional[str]: Path to the generated preview image or None if failed
        """
        try:
            # Log that we're generating a preview
            self.write_debug(
                "=== GENERATING MODEL PREVIEW ===\n",
                f"Filename: {filename}.png\n",
                f"Resolution: {resolution}\n",
                f"Camera parameters: {camera_params}\n",
                f"Image size: {width}x{height}\n",
                "=" * 50 + "\n\n"
            )
            
            # Use the model exporter to generate the preview
            preview_path = self.model_exporter.generate_preview(
                scad_code=scad_code,
                filename=filename,
                resolution=resolution,
                camera_params=camera_params,
                width=width,
                height=height
            )
            
            if preview_path:
                self.write_debug(
                    "Preview generated successfully: " + preview_path + "\n\n"
                )
            else:
                self.write_debug(
                    "Failed to generate preview\n\n"
                )
                
            return preview_path
        except Exception as e:
            error_msg = f"Error generating model preview: {str(e)}"
            print(f"\n{error_msg}")
            self.write_debug(
                "=== PREVIEW GENERATION ERROR ===\n",
                f"Error: {error_msg}\n",
                "=" * 50 + "\n\n"
            )
            return None
    
    def get_preview_settings(self):
        """Get user input for preview settings.
        
        Returns:
            dict: Preview settings including resolution, camera parameters, and image dimensions
        """
        print("\nPreview Settings:")
        print("-" * 30)
        
        # Get resolution
        resolution_input = input("Enter resolution value ($fn) [default: 100]: ").strip()
        resolution = int(resolution_input) if resolution_input.isdigit() else 100
        
        # Ask if user wants to use custom camera settings
        custom_camera = input("Do you want to use custom camera settings? (y/n) [default: n]: ").lower().strip() == 'y'
        
        if custom_camera:
            print("\nCamera settings are in format: translateX,translateY,translateZ,rotateX,rotateY,rotateZ,distance")
            print("Default is: 0,0,0,55,0,25,140")
            
            camera_input = input("Enter camera parameters: ").strip()
            # Validate camera input format (should be 7 comma-separated numbers)
            if camera_input and len(camera_input.split(',')) == 7 and all(part.replace('-', '', 1).replace('.', '', 1).isdigit() for part in camera_input.split(',')):
                camera_params = camera_input
            else:
                print("Invalid camera parameters. Using default settings.")
                camera_params = "0,0,0,55,0,25,140"
        else:
            camera_params = "0,0,0,55,0,25,140"
        
        # Get image dimensions
        custom_dimensions = input("Do you want to use custom image dimensions? (y/n) [default: n]: ").lower().strip() == 'y'
        
        if custom_dimensions:
            width_input = input("Enter image width (pixels) [default: 800]: ").strip()
            width = int(width_input) if width_input.isdigit() and int(width_input) > 0 else 800
            
            height_input = input("Enter image height (pixels) [default: 600]: ").strip()
            height = int(height_input) if height_input.isdigit() and int(height_input) > 0 else 600
        else:
            width = 800
            height = 600
        
        return {
            "resolution": resolution,
            "camera_params": camera_params,
            "width": width,
            "height": height
        }

    def export_to_multiple_formats(self, scad_code, formats=["stl"], filename="model", resolution=100, additional_params=None):
        """Export SCAD code to multiple formats
        
        Args:
            scad_code: OpenSCAD code to export
            formats: List of formats to export to
            filename: Base name for output files
            resolution: Resolution for curves
            additional_params: Additional parameters for OpenSCAD
            
        Returns:
            Dictionary mapping formats to ExportResult objects
        """
        results = {}
        
        for fmt in formats:
            results[fmt] = self.export_model(
                scad_code=scad_code,
                format=fmt,
                filename=filename,
                resolution=resolution,
                additional_params=additional_params
            )
            
        return results
    
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
                
                examples, extracted_metadata = self.retrieve_similar_examples(
                    description, step_back_result, keyword_data
                )
                
                # Step 3.5: Parameter customization (now handled by the graph, but keep step for completeness)
                print("\n" + "="*50)
                print("STEP 3.5: PARAMETER CUSTOMIZATION")
                print("="*50)
                print("Parameter identification is handled by the graph system...")
                
                # Step 4: Generate OpenSCAD code
                print("\n" + "="*50)
                print("STEP 4: GENERATING SCAD CODE")
                print("="*50)
                
                result = self.generate_scad_code(description, examples, step_back_result, {})
                if not result['success']:
                    error_msg = result['error']
                    print(f"\nError: {error_msg}")
                    self.write_debug(
                        "=== GENERATION ERROR ===\n",
                        f"Error: {error_msg}\n",
                        "=" * 50 + "\n\n"
                    )
                    
                    # Provide more detailed feedback and suggestions
                    print("\nSuggestions to improve your request:")
                    print("1. Be more specific about dimensions, shape, and features")
                    print("2. Specify the purpose or function of the object")
                    print("3. Mention any similar real-world objects it should resemble")
                    print("4. Include details about materials or textures if relevant")
                    print("5. Describe how different parts connect or interact")
                    
                    # Ask if user wants to try again with a more detailed description
                    retry_with_details = input("\nWould you like to try again with a more detailed description? (y/n): ").lower().strip()
                    if retry_with_details == 'y':
                        new_description = input("\nPlease provide a more detailed description: ")
                        # Recursively call generate_model with the new description
                        return self.generate_model(new_description)
                    
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
                
                # Ask user if they want to preview the model
                preview_model = input("\nWould you like to generate a preview of this model? (y/n): ").lower().strip()
                
                if preview_model == 'y':
                    # Get custom preview settings
                    preview_settings = self.get_preview_settings()
                    
                    print("\nGenerating preview...")
                    preview_path = self.generate_model_preview(
                        scad_code=scad_code,
                        filename="preview",
                        resolution=preview_settings["resolution"],
                        camera_params=preview_settings["camera_params"],
                        width=preview_settings["width"],
                        height=preview_settings["height"]
                    )
                    
                    if preview_path and os.path.exists(preview_path):
                        print(f"\nPreview generated successfully: {preview_path}")
                        
                        # Try to open the preview image with the default viewer
                        try:
                            if os.name == 'posix':  # macOS or Linux
                                if os.path.exists('/usr/bin/open'):  # macOS
                                    subprocess.run(["open", preview_path])
                                else:  # Linux
                                    subprocess.run(["xdg-open", preview_path])
                            elif os.name == 'nt':  # Windows
                                os.startfile(preview_path)
                            print("\nOpened preview in default image viewer.")
                        except Exception as e:
                            print(f"\nCould not automatically open preview: {e}")
                            print(f"Please open the preview image located at: {preview_path}")
                    else:
                        print("\nFailed to generate preview. Please check the logs for details.")
                
                # Ask user if they want to tune the parameters
                tune_params = input("\nWould you like to tune the parameters for this model? (y/n): ").lower().strip()
                
                if tune_params == 'y':
                    print("\nTuning parameters...")
                    tuning_result = self.parameter_tuner.tune_parameters(scad_code, description)
                    
                    if tuning_result.get("success") and tuning_result.get("adjustments"):
                        scad_code = tuning_result.get("updated_code")
                        print("\nParameters tuned successfully!")
                        
                        # Save updated code to file
                        with open("output.scad", "w") as f:
                            f.write(scad_code)
                        
                        print("\nUpdated OpenSCAD code has been saved to 'output.scad'")
                    else:
                        print("\nNo parameter changes were applied.")
                
                # Ask user if they want to export this model to other formats
                export_model = input("\nWould you like to export this model to other formats? (y/n): ").lower().strip()
                exported_files = None
                
                if export_model == 'y':
                    print("\nAvailable export formats: stl, off, amf, 3mf, csg, dxf, svg, png")
                    formats_input = input("Enter comma-separated formats to export (default: stl): ").strip()
                    formats = [f.strip() for f in formats_input.split(',')] if formats_input else ["stl"]
                    
                    resolution = input("Enter resolution value (default: 100): ").strip()
                    resolution = int(resolution) if resolution.isdigit() else 100
                    
                    filename = input("Enter base filename (default: model): ").strip() or "model"
                    
                    print("\nExporting model to selected formats...")
                    exported_files = self.export_to_multiple_formats(
                        scad_code=scad_code,
                        formats=formats,
                        filename=filename,
                        resolution=resolution
                    )
                    
                    # Print export results
                    print("\nExport Results:")
                    for fmt, result in exported_files.items():
                        if result.success:
                            print(f"  - {fmt}: Success! Saved to {result.file_path}")
                        else:
                            print(f"  - {fmt}: Failed - {result.error}")
                
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
                
                if hasattr(self, 'session_integration'):
                    metadata = {
                        "complexity": extracted_metadata.get("complexity", "medium") if extracted_metadata else "medium",
                        "category": extracted_metadata.get("category", "unknown") if extracted_metadata else "unknown",
                        "techniques": extracted_metadata.get("techniques_used", []) if extracted_metadata else [],
                        "added_to_kb": add_to_kb == 'y'
                    }
                    self.session_integration.record_generation(description, scad_code, metadata)
                
                # Save complete debug log
                self.save_debug_log()
        
                return {
                    'success': True,
                    'code': scad_code,
                    'exported_files': {fmt: result.model_dump() for fmt, result in exported_files.items()} if exported_files else None,
                    'parameters_tuned': tune_params == 'y'
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