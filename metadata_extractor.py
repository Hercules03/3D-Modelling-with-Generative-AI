from llm_management import LLMProvider, ModelDefinitions
import json
from datetime import datetime
from prompts import METADATA_EXTRACTION_PROMPT, KEYWORD_EXTRACTOR_PROMPT, CATEGORY_ANALYSIS_PROMPT
from conversation_logger import ConversationLogger
from LLMPromptLogger import LLMPromptLogger
from constant import BASIC_KNOWLEDGE

class MetadataExtractor:
    def __init__(self, llm_provider="gemma"):
        """Initialize the metadata extractor with an LLM provider"""
        print("\nInitializing Metadata Extractor...")
        # Main LLM for metadata extraction (using Gemma)
        print("- Using gemma3:4b-it-q8_0 for metadata extraction")
        self.llm = LLMProvider.get_llm(
            provider="gemma",
            temperature=0.7,
            model=ModelDefinitions.GEMMA  # Explicitly use 4b model for main extraction
        )
        print("- Main LLM provider initialized")
        
        self.extraction_prompt = METADATA_EXTRACTION_PROMPT
        self.logger = ConversationLogger()
        self.prompt_logger = LLMPromptLogger()  # Add LLMPromptLogger

    def extract_metadata(self, description, code="", step_back_result=None, keyword_data=None):
        """Extract metadata from a model description and code"""
        try:
            print(f"\nExtracting metadata")
            
            timestamp = datetime.now().isoformat()
            
            # Handle case where keyword_data is None
            if keyword_data is None:
                print("No keyword data provided, using default values")
                keyword_data = {
                    "core_type": "model",
                    "modifiers": [],
                    "compound_type": ""
                }
            # Also handle string type conversion (in case it's not a dictionary)
            elif isinstance(keyword_data, str):
                print("Converting string keyword_data to dictionary")
                try:
                    keyword_data = json.loads(keyword_data)
                except json.JSONDecodeError:
                    keyword_data = {
                        "core_type": "model",
                        "modifiers": [],
                        "compound_type": ""
                    }
            
            # Create the analysis object for keyword extraction
            analysis = {
                "query": {
                    "input": description,
                    "timestamp": timestamp,
                    "model": ModelDefinitions.KEYWORD_EXTRACTOR
                },
                "response": {
                    "core_type": keyword_data.get("core_type", ""),
                    "modifiers": keyword_data.get("modifiers", []),
                    "compound_type": keyword_data.get("compound_type", "")
                },
                "metadata": {
                    "success": True,
                    "error": None
                }
            }
            print(f"Created analysis object: {json.dumps(analysis, indent=2)}")
            
            # Use the extracted type for metadata
            object_type = (keyword_data.get("compound_type") or 
                         keyword_data.get("core_type", ""))
            print(f"Using object type: {object_type}")
            
            # Get LLM-based metadata
            print("\n=== Starting Full Metadata Extraction ===")
            prompt = self.extraction_prompt.format(description=description, step_back_analysis=step_back_result)
            print("Invoking main LLM for metadata extraction...")
            response = self.llm.invoke(prompt)
            
            # Parse JSON response for full metadata
            try:
                # Clean up the response by removing code block markers
                clean_content = response.content.replace("```json", "").replace("```", "").strip()
                metadata = json.loads(clean_content)
                # Update with our enhanced object type
                metadata["object_type"] = object_type
                
                # Add code analysis metadata if code is provided
                if code:
                    code_metadata = self.analyze_code_metadata(code)
                    metadata.update(code_metadata)
                
                # Ensure required fields exist
                metadata.setdefault('features', [])
                metadata.setdefault('geometric_properties', [])
                metadata.setdefault('materials', [])
                metadata.setdefault('technical_requirements', [])
                metadata.setdefault('complexity', 'SIMPLE')
                metadata.setdefault('style', 'Modern')
                metadata.setdefault('use_case', [])
                
                print("Successfully extracted metadata:")
                for key, value in metadata.items():
                    print(f"  • {key}: {value}")
                
                # Log metadata extraction
                self.prompt_logger.log_metadata_extraction(
                    query=description,
                    code=code,
                    response=metadata,
                    timestamp=timestamp
                )
                
                # Perform category analysis
                category_result = self.analyze_categories(description, metadata)
                if category_result:
                    metadata.update(category_result)
                    self.prompt_logger.log_category_analysis(
                        query=description,
                        code=code,
                        response=category_result,
                        timestamp=timestamp
                    )
                
                print("=== Full Metadata Extraction Complete ===\n")
                return metadata
                
            except json.JSONDecodeError as e:
                print(f"Error parsing metadata response as JSON: {e}")
                print(f"Raw response that failed to parse: {response.content}")
                print("=== Full Metadata Extraction Failed ===\n")
                return {"object_type": object_type}
                
        except Exception as e:
            print(f"Metadata extraction failed with error: {e}")
            print("Stack trace:")
            import traceback
            traceback.print_exc()
            print("=== Metadata Extraction Failed ===\n")
            return {}

    def analyze_categories(self, description, metadata):
        """Analyze categories for the given description and metadata"""
        try:
            # Format the prompt with standard categories and properties
            prompt = CATEGORY_ANALYSIS_PROMPT.format(
                object_type=metadata.get("object_type", ""),
                description=description,
                categories_info=BASIC_KNOWLEDGE["categories"],
                properties_info=BASIC_KNOWLEDGE["properties"]
            )
            
            # Get category analysis from LLM
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            try:
                # Clean up and parse JSON response
                clean_content = content.replace("```json", "").replace("```", "").strip()
                category_data = json.loads(clean_content)
                return category_data
            except json.JSONDecodeError as e:
                print(f"Error parsing category analysis response: {e}")
                return None
                
        except Exception as e:
            print(f"Category analysis failed: {e}")
            return None

    def analyze_code_metadata(self, code: str) -> dict:
        """Analyze OpenSCAD code to extract additional metadata"""
        metadata = {}
        
        try:
            # Calculate complexity score
            complexity_score = self._calculate_complexity_score(code)
            if complexity_score < 30:
                metadata["complexity"] = "SIMPLE"
            elif complexity_score < 70:
                metadata["complexity"] = "MEDIUM"
            else:
                metadata["complexity"] = "COMPLEX"
            
            # Analyze components
            components = self._analyze_components(code)
            metadata["components"] = components
            
            # Extract geometric properties
            geometric_props = []
            if any(c["name"] == "sphere" for c in components):
                geometric_props.append("spherical")
            if any(c["name"] == "cube" for c in components):
                geometric_props.append("angular")
            if any(c["name"] == "cylinder" for c in components):
                geometric_props.append("cylindrical")
            if any(c["type"] == "boolean" for c in components):
                geometric_props.append("compound")
            metadata["geometric_properties"] = geometric_props
            
            # Analyze technical requirements
            tech_reqs = []
            if len([c for c in components if c["type"] == "boolean"]) > 2:
                tech_reqs.append("complex_boolean_operations")
            if len([c for c in components if c["type"] == "transformation"]) > 3:
                tech_reqs.append("multiple_transformations")
            if any(c["type"] == "module" for c in components):
                tech_reqs.append("modular_design")
            metadata["technical_requirements"] = tech_reqs
            
            return metadata
            
        except Exception as e:
            print(f"Error analyzing code metadata: {str(e)}")
            return {}

    def _calculate_complexity_score(self, code: str) -> float:
        """Calculate complexity score based on code analysis"""
        score = 0
        
        try:
            # Count number of operations/functions
            operations = len([line for line in code.split('\n') 
                            if any(op in line.lower() for op in ['union', 'difference', 'intersection', 'translate', 'rotate', 'scale'])])
            score += min(operations * 2, 30)  # Max 30 points for operations
            
            # Count nested levels
            max_nesting = 0
            current_nesting = 0
            for line in code.split('\n'):
                if '{' in line:
                    current_nesting += 1
                    max_nesting = max(max_nesting, current_nesting)
                if '}' in line:
                    current_nesting -= 1
            score += min(max_nesting * 5, 20)  # Max 20 points for nesting
            
            # Count variables and modules
            variables = len([line for line in code.split('\n') if '=' in line and not line.strip().startswith('//')])
            modules = len([line for line in code.split('\n') if 'module' in line])
            score += min((variables + modules) * 2, 20)  # Max 20 points for variables/modules
            
            # Analyze geometric complexity
            geometric_ops = len([line for line in code.split('\n') 
                               if any(shape in line.lower() for shape in ['sphere', 'cube', 'cylinder', 'polyhedron'])])
            score += min(geometric_ops * 3, 15)  # Max 15 points for geometric operations
            
            return min(max(score, 0), 100)
            
        except Exception as e:
            print(f"Error calculating code complexity: {str(e)}")
            return 0

    def _analyze_components(self, code: str) -> list:
        """Analyze and extract components from SCAD code"""
        components = []
        try:
            import re
            
            # Extract modules (reusable components)
            module_pattern = r'module\s+(\w+)\s*\([^)]*\)\s*{'
            modules = re.findall(module_pattern, code)
            for module in modules:
                components.append({
                    'type': 'module',
                    'name': module,
                    'reusable': True
                })
            
            # Extract main geometric shapes
            shape_pattern = r'(sphere|cube|cylinder|polyhedron)\s*\('
            shapes = re.findall(shape_pattern, code)
            for shape in shapes:
                components.append({
                    'type': 'primitive',
                    'name': shape,
                    'reusable': False
                })
            
            # Extract transformations
            transform_pattern = r'(translate|rotate|scale|mirror)\s*\('
            transformations = re.findall(transform_pattern, code)
            for transform in transformations:
                components.append({
                    'type': 'transformation',
                    'name': transform,
                    'reusable': False
                })
            
            # Extract boolean operations
            bool_pattern = r'(union|difference|intersection)\s*\('
            booleans = re.findall(bool_pattern, code)
            for boolean in booleans:
                components.append({
                    'type': 'boolean',
                    'name': boolean,
                    'reusable': False
                })
            
            return components
            
        except Exception as e:
            print(f"Error analyzing components: {str(e)}")
            return [] 