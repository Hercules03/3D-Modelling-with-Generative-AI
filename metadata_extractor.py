from llm_management import LLMProvider, ModelDefinitions
import json
from datetime import datetime
from prompts import METADATA_EXTRACTION_PROMPT, KEYWORD_EXTRACTOR_PROMPT, CATEGORY_ANALYSIS_PROMPT
from conversation_logger import ConversationLogger
from LLMPromptLogger import LLMPromptLogger
from constant import BASIC_KNOWLEDGE
import re
import logging

logger = logging.getLogger(__name__)

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
        self.prompt_logger = LLMPromptLogger()

    def extract_metadata(self, description, code="", step_back_result=None, keyword_data=None):
        """
        Extract metadata from a model description and code.
        
        Args:
            description: The model description
            code: Optional OpenSCAD code
            step_back_result: Optional step-back analysis results
            keyword_data: Optional pre-extracted keyword data
            
        Returns:
            Dictionary containing extracted metadata
        """
        try:
            print(f"\nExtracting metadata")
            
            timestamp = datetime.now().isoformat()
            
            # Process keyword data
            keyword_data = self._process_keyword_data(keyword_data)
            
            # Create analysis object
            analysis = self._create_analysis_object(description, timestamp, keyword_data)
            
            # Get object type from keyword data
            object_type = (keyword_data.get("compound_type") or 
                         keyword_data.get("core_type", ""))
            print(f"Using object type: {object_type}")
            
            # Extract base metadata using LLM
            metadata = self._extract_base_metadata(description, step_back_result)
            if not metadata:
                metadata = {"object_type": object_type}
            
            # Add code metadata if code is provided
            if code:
                code_metadata = self.analyze_code_metadata(code)
                metadata.update(code_metadata)
            
            # Ensure required fields exist
            metadata = self._ensure_required_fields(metadata)
            
            # Update with object type from keyword data
            metadata["object_type"] = object_type
            
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
            
            # Log the extraction
            self.prompt_logger.log_metadata_extraction(
                query=description,
                code=code,
                response=metadata,
                timestamp=timestamp
            )
            
            print("=== Full Metadata Extraction Complete ===\n")
            return metadata
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return {}

    def _process_keyword_data(self, keyword_data):
        """Process and validate keyword data"""
        if keyword_data is None:
            return {
                "core_type": "model",
                "modifiers": [],
                "compound_type": ""
            }
        
        if isinstance(keyword_data, str):
            try:
                return json.loads(keyword_data)
            except json.JSONDecodeError:
                return {
                    "core_type": "model",
                    "modifiers": [],
                    "compound_type": ""
                }
        
        return keyword_data

    def _create_analysis_object(self, description, timestamp, keyword_data):
        """Create an analysis object for logging"""
        return {
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

    def _extract_base_metadata(self, description, step_back_result):
        """Extract base metadata using LLM"""
        try:
            prompt = self.extraction_prompt.format(
                description=description, 
                step_back_analysis=step_back_result
            )
            response = self.llm.invoke(prompt)
            
            # Clean up and parse response
            clean_content = response.content.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_content)
            
        except Exception as e:
            logger.error(f"Base metadata extraction failed: {str(e)}")
            return None

    def _ensure_required_fields(self, metadata):
        """Ensure all required metadata fields exist with default values"""
        defaults = {
            'features': [],
            'geometric_properties': [],
            'materials': [],
            'dimensions': {},
            'use_case': [],
            'technical_requirements': [],
            'complexity': 'SIMPLE',
            'style': 'Modern',
            'step_back_analysis': {
                'core_principles': [],
                'shape_components': [],
                'implementation_steps': []
            }
        }
        
        for key, default_value in defaults.items():
            metadata.setdefault(key, default_value)
        
        return metadata

    def analyze_code_metadata(self, code):
        """
        Analyze OpenSCAD code to extract metadata about complexity,
        components, and technical requirements.
        """
        try:
            metadata = {}
            
            # Calculate complexity score and level
            complexity_score = self._calculate_complexity_score(code)
            metadata['complexity_score'] = complexity_score
            if complexity_score < 30:
                metadata['complexity'] = 'SIMPLE'
            elif complexity_score < 70:
                metadata['complexity'] = 'MEDIUM'
            else:
                metadata['complexity'] = 'COMPLEX'
            
            # Analyze components
            components = self._analyze_components(code)
            metadata['components'] = components
            
            # Extract geometric properties
            geometric_props = self._extract_geometric_properties(components)
            metadata['geometric_properties'] = geometric_props
            
            # Analyze technical requirements
            tech_reqs = self._analyze_technical_requirements(components)
            metadata['technical_requirements'] = tech_reqs
            
            return metadata
            
        except Exception as e:
            logger.error(f"Code metadata analysis failed: {str(e)}")
            return {}

    def _calculate_complexity_score(self, code):
        """Calculate complexity score based on code analysis"""
        score = 0
        
        try:
            # Count operations/functions
            operations = len([line for line in code.split('\n') 
                            if any(op in line.lower() for op in [
                                'union', 'difference', 'intersection',
                                'translate', 'rotate', 'scale'
                            ])])
            score += min(operations * 2, 30)  # Max 30 points
            
            # Count nesting levels
            max_nesting = 0
            current_nesting = 0
            for line in code.split('\n'):
                if '{' in line:
                    current_nesting += 1
                    max_nesting = max(max_nesting, current_nesting)
                if '}' in line:
                    current_nesting -= 1
            score += min(max_nesting * 5, 20)  # Max 20 points
            
            # Count variables and modules
            variables = len([line for line in code.split('\n') 
                           if '=' in line and not line.strip().startswith('//')])
            modules = len([line for line in code.split('\n') 
                         if 'module' in line])
            score += min((variables + modules) * 2, 20)  # Max 20 points
            
            # Count geometric operations
            geometric_ops = len([line for line in code.split('\n') 
                               if any(shape in line.lower() for shape in [
                                   'sphere', 'cube', 'cylinder', 'polyhedron'
                               ])])
            score += min(geometric_ops * 3, 15)  # Max 15 points
            
        except Exception as e:
            logger.error(f"Error calculating complexity score: {str(e)}")
        
        return min(max(score, 0), 100)  # Normalize to 0-100

    def _analyze_components(self, code):
        """Analyze and extract components from OpenSCAD code"""
        components = []
        try:
            # Extract modules
            module_pattern = r'module\s+(\w+)\s*\([^)]*\)\s*{'
            modules = re.findall(module_pattern, code)
            for module in modules:
                components.append({
                    'type': 'module',
                    'name': module,
                    'reusable': True
                })
            
            # Extract geometric shapes
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
            
        except Exception as e:
            logger.error(f"Error analyzing components: {str(e)}")
        
        return components

    def _extract_geometric_properties(self, components):
        """Extract geometric properties based on components"""
        properties = []
        
        try:
            # Analyze basic shapes
            if any(c["name"] == "sphere" for c in components):
                properties.append("spherical")
            if any(c["name"] == "cube" for c in components):
                properties.append("angular")
            if any(c["name"] == "cylinder" for c in components):
                properties.append("cylindrical")
            
            # Analyze complexity
            if any(c["type"] == "boolean" for c in components):
                properties.append("compound")
            
            # Analyze symmetry
            if any(c["name"] == "mirror" for c in components):
                properties.append("symmetric")
            
            # Analyze modularity
            if any(c["type"] == "module" for c in components):
                properties.append("modular")
                
        except Exception as e:
            logger.error(f"Error extracting geometric properties: {str(e)}")
        
        return properties

    def _analyze_technical_requirements(self, components):
        """Analyze technical requirements based on components"""
        requirements = []
        
        try:
            # Analyze boolean operations
            bool_ops = [c for c in components if c["type"] == "boolean"]
            if len(bool_ops) > 2:
                requirements.append("complex_boolean_operations")
            
            # Analyze transformations
            transforms = [c for c in components if c["type"] == "transformation"]
            if len(transforms) > 3:
                requirements.append("multiple_transformations")
            
            # Analyze modularity
            if any(c["type"] == "module" for c in components):
                requirements.append("modular_design")
            
            # Analyze shape complexity
            if len([c for c in components if c["type"] == "primitive"]) > 3:
                requirements.append("complex_geometry")
                
        except Exception as e:
            logger.error(f"Error analyzing technical requirements: {str(e)}")
        
        return requirements

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
                logger.error(f"Error parsing category analysis response: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Category analysis failed: {str(e)}")
            return None 