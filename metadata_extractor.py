from LLM import LLMProvider, ModelDefinitions
import json
from datetime import datetime
from prompts import METADATA_EXTRACTION_PROMPT, KEYWORD_EXTRACTOR_PROMPT, CATEGORY_ANALYSIS_PROMPT
from constant import BASIC_KNOWLEDGE
import re
from typing import Dict, List, Union, Optional
import logging
import traceback
from fuzzywuzzy import fuzz

logger = logging.getLogger(__name__)

class MetadataExtractor:
    # Add standard categories as a class variable
    STANDARD_CATEGORIES = {
        "container": {
            "description": "Objects that can hold or contain other items",
            "examples": ["cup", "mug", "bowl", "vase", "bottle", "box", "jar", "tray", "basket", "planter", "container"]
        },
        "furniture": {
            "description": "Objects used for living or working spaces",
            "examples": ["table", "chair", "desk", "shelf", "cabinet", "bench", "stand", "organizer", "rack", "holder"]
        },
        "decorative": {
            "description": "Objects primarily for aesthetic purposes",
            "examples": ["sculpture", "ornament", "statue", "figurine", "vase", "frame", "display", "art", "decoration"]
        },
        "functional": {
            "description": "Utility objects with specific practical purposes",
            "examples": ["holder", "stand", "bracket", "hook", "hanger", "clip", "mount", "support", "organizer", "adapter"]
        },
        "geometric": {
            "description": "Basic or complex geometric shapes and mathematical objects",
            "examples": ["cube", "sphere", "cylinder", "cone", "polyhedron", "torus", "prism", "pyramid", "helix"]
        },
        "mechanical": {
            "description": "Parts or components of mechanical systems",
            "examples": ["gear", "bolt", "nut", "bearing", "lever", "hinge", "joint", "wheel", "pulley", "spring"]
        },
        "enclosure": {
            "description": "Objects designed to enclose or protect other items",
            "examples": ["case", "box", "housing", "cover", "shell", "enclosure", "protection", "sleeve", "cap"]
        },
        "modular": {
            "description": "Objects designed to be combined or connected with others",
            "examples": ["connector", "adapter", "joint", "mount", "tile", "block", "module", "segment", "piece"]
        },
        "other": {
            "description": "Objects that don't fit in other categories",
            "examples": []
        }
    }

    STANDARD_PROPERTIES = {
        # Physical dimensions and characteristics
        "size": ["tiny", "small", "medium", "large", "huge"],
        "wall_thickness": ["thin", "medium", "thick", "solid"],
        "hollow": ["yes", "no", "partial"],
        
        # Design characteristics
        "style": ["modern", "traditional", "minimalist", "decorative", "industrial", "organic", "geometric", "artistic"],
        "complexity": ["simple", "moderate", "complex", "intricate"],
        "symmetry": ["radial", "bilateral", "asymmetric", "periodic"],
        
        # 3D Printing specific
        "printability": ["easy", "moderate", "challenging", "requires_support"],
        "orientation": ["flat", "vertical", "angled", "any"],
        "support_needed": ["none", "minimal", "moderate", "extensive"],
        "infill_requirement": ["low", "medium", "high", "solid"],
        
        # Assembly and structure
        "assembly": ["single_piece", "multi_part", "snap_fit", "threaded", "interlocking"],
        "stability": ["self_standing", "needs_support", "wall_mounted", "hanging"],
        "adjustability": ["fixed", "adjustable", "modular", "customizable"],
        
        # Material and finishing
        "material_compatibility": ["any", "pla", "abs", "petg", "resin", "multi_material"],
        "surface_finish": ["smooth", "textured", "patterned", "functional"],
        "post_processing": ["none", "minimal", "required", "optional"],
        
        # Functional aspects
        "durability": ["light_duty", "medium_duty", "heavy_duty"],
        "water_resistance": ["none", "splash_proof", "water_tight"],
        "heat_resistance": ["low", "medium", "high"],
        
        # Design intent
        "customization": ["fixed", "parametric", "highly_customizable"],
        "purpose": ["practical", "decorative", "educational", "prototype"]
    }

    def __init__(self, llm_provider, conversation_logger, prompt_logger):
        """Initialize the metadata extractor with an LLM provider"""
        logger.info("Initializing Metadata Extractor...")
        print("\nInitializing Metadata Extractor...")
        # Main LLM for metadata extraction
        print(f"- Using {llm_provider} for metadata extraction")
        # Get the appropriate model based on the provider
        model = None
        if llm_provider == "anthropic":
            model = ModelDefinitions.ANTHROPIC
        elif llm_provider == "openai":
            model = ModelDefinitions.OPENAI
        elif llm_provider == "gemma":
            model = ModelDefinitions.GEMMA
        elif llm_provider == "deepseek":
            model = ModelDefinitions.DEEPSEEK
        
        self.llm = LLMProvider.get_llm(
            provider=llm_provider,
            temperature=0.7,
            model=model
        )
        print("- Metadata Extractor LLM provider initialized")
        
        self.extraction_prompt = METADATA_EXTRACTION_PROMPT
        self.logger = conversation_logger
        self.prompt_logger = prompt_logger
        
        # Define required metadata fields
        self.required_fields = {
            'object_type': str,
            'features': list,
            'geometric_properties': list,
            'materials': list,
            'technical_requirements': list,
            'complexity': str,
            'style': str,
            'use_case': list,
            'categories': list,
            'properties': dict
        }

    def extract_metadata(self, description, code="", step_back_result=None, keyword_data=None):
        """
        Extract metadata from a model description and code.
        This is the main entry point for metadata extraction.
        """
        try:
            print(f"\nExtracting metadata")
            timestamp = datetime.now().isoformat()
            
            # 1. Process keyword data
            keyword_data = self._process_keyword_data(keyword_data)
            object_type = (keyword_data.get("compound_type") or keyword_data.get("core_type", ""))
            
            # 2. Extract base metadata from description and step-back analysis
            base_metadata = self._extract_base_metadata(description, step_back_result)
            if not base_metadata:
                base_metadata = {}
            
            # 3. Add code analysis metadata if code is provided
            if code:
                code_metadata = self.analyze_code_metadata(code)
                base_metadata.update(code_metadata)
            
            # 4. Add keyword-based metadata
            base_metadata["object_type"] = object_type
            
            # 5. Perform category analysis
            category_result = self.analyze_categories(description, base_metadata)
            if category_result:
                base_metadata.update(category_result)
            
            # 6. Ensure all required fields exist with correct types
            metadata = self._validate_and_normalize_metadata(base_metadata)
            
            # 7. Log the extraction
            self._log_extraction(description, code, metadata, timestamp)
            
            print("=== Full Metadata Extraction Complete ===\n")
            return metadata
            
        except Exception as e:
            print(f"Metadata extraction failed with error: {e}")
            import traceback
            traceback.print_exc()
            print("=== Metadata Extraction Failed ===\n")
            return self._get_default_metadata()

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

    def _extract_base_metadata(self, description, step_back_result):
        """Extract base metadata using LLM"""
        try:
            prompt = self.extraction_prompt.format(
                description=description,
                step_back_analysis=step_back_result
            )
            response = self.llm.invoke(prompt)
            clean_content = response.content.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_content)
        except Exception as e:
            print(f"Error extracting base metadata: {e}")
            return None

    def _validate_and_normalize_metadata(self, metadata):
        """Ensure all required fields exist with correct types"""
        normalized = {}
        
        for field, field_type in self.required_fields.items():
            value = metadata.get(field)
            
            # Convert to correct type if needed
            if value is None:
                if field_type == list:
                    value = []
                elif field_type == dict:
                    value = {}
                elif field_type == str:
                    value = ""
            elif isinstance(value, str):
                if field_type == list:
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        value = [value] if value else []
                elif field_type == dict:
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        value = {}
            
            normalized[field] = value
        
        return normalized

    def _get_default_metadata(self):
        """Return default metadata structure"""
        return {
            'object_type': '',
            'features': [],
            'geometric_properties': [],
            'materials': [],
            'technical_requirements': [],
            'complexity': 'SIMPLE',
            'style': 'Modern',
            'use_case': [],
            'categories': ['other'],
            'properties': {}
        }

    def _log_extraction(self, description, code, metadata, timestamp):
        """Log metadata extraction results"""
        self.prompt_logger.log_metadata_extraction(
            query=description,
            code=code,
            response=metadata,
            timestamp=timestamp
        )

    def analyze_categories(self, description, metadata=None):
        """
        Analyze categories for the given description and metadata.
        This is the centralized method for category analysis.
        """
        try:
            if metadata is None:
                metadata = {'object_type': ''}
            
            # Prepare category info from standard categories
            categories_info = BASIC_KNOWLEDGE.get("categories")
            if not categories_info:
                categories_info = "\n".join(
                    f"- {cat}: {info['description']} (e.g., {', '.join(info['examples'])})"
                    for cat, info in self.STANDARD_CATEGORIES.items()
                )
            
            # Prepare properties info from standard properties
            properties_info = BASIC_KNOWLEDGE.get("properties")
            if not properties_info:
                properties_info = "\n".join(
                    f"- {prop}: {', '.join(values)}"
                    for prop, values in self.STANDARD_PROPERTIES.items()
                )
            
            # Format the prompt with categories and properties
            prompt = CATEGORY_ANALYSIS_PROMPT.format(
                object_type=metadata.get("object_type", ""),
                description=description,
                categories_info=categories_info,
                properties_info=properties_info
            )
            
            # Get category analysis from LLM
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            try:
                # Clean up and parse JSON response
                clean_content = content.replace("```json", "").replace("```", "").strip()
                category_data = json.loads(clean_content)
                
                # Validate and clean the response if using standard categories
                if hasattr(self, 'STANDARD_CATEGORIES'):
                    cleaned_result = {
                        "categories": [
                            cat for cat in category_data.get("categories", [])
                            if cat in self.STANDARD_CATEGORIES
                        ],
                        "properties": {
                            prop: value
                            for prop, value in category_data.get("properties", {}).items()
                            if prop in self.STANDARD_PROPERTIES
                            and value in self.STANDARD_PROPERTIES[prop]
                        },
                        "similar_objects": [
                            obj for obj in category_data.get("similar_objects", [])
                            if any(obj in cat_info["examples"]
                                for cat_info in self.STANDARD_CATEGORIES.values())
                        ]
                    }
                    
                    # Ensure at least one category is assigned
                    if not cleaned_result["categories"]:
                        cleaned_result["categories"] = ["container"]  # Default to container for most 3D objects
                    
                    return cleaned_result
                
                return category_data
            except json.JSONDecodeError as e:
                print(f"Error parsing category analysis response: {e}")
                return {
                    "categories": ["other"],
                    "properties": {},
                    "similar_objects": []
                }
                
        except Exception as e:
            print(f"Category analysis failed: {e}")
            return {
                "categories": ["other"],
                "properties": {},
                "similar_objects": []
            }

    def analyze_code_metadata(self, code: str) -> dict:
        """
        Analyze OpenSCAD code to extract additional metadata.
        This is the centralized method for code analysis.
        """
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
            
            # Group components by type
            grouped_components = self._group_components_by_type(components)
            metadata["grouped_components"] = grouped_components
            
            return metadata
            
        except Exception as e:
            print(f"Error analyzing code metadata: {str(e)}")
            return {}

    def _calculate_complexity_score(self, code: str) -> float:
        """
        Calculate complexity score based on code analysis.
        Centralized method for complexity scoring.
        """
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
        """
        Analyze and extract components from SCAD code.
        Centralized method for component extraction.
        """
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

    def _group_components_by_type(self, components: List[Dict]) -> Dict[str, List[Dict]]:
        """Group components by their type"""
        grouped = {}
        for component in components:
            if isinstance(component, dict) and 'type' in component and 'name' in component:
                comp_type = component['type'].lower()
                if comp_type not in grouped:
                    grouped[comp_type] = []
                grouped[comp_type].append(component)
        return grouped
    
    def calculate_text_similarity(self, list1, list2):
        """
        Calculate similarity between two lists of text items using semantic matching.
        """
        if not list1 or not list2:
            return 0.0
            
        # Convert to lists if strings
        if isinstance(list1, str):
            list1 = [list1]
        if isinstance(list2, str):
            list2 = [list2]
        
        # Keywords that indicate mechanical/automotive relevance
        mechanical_keywords = {
            'circular', 'cylindrical', 'rotational', 'symmetric', 'concentric',
            'wheel', 'rim', 'hub', 'spoke', 'bolt', 'mounting', 'automotive',
            'vehicle', 'car', 'truck', 'axle', 'bearing', 'blade', 'propeller'
        }
        
        # Object type pairs with high similarity
        object_type_similarities = {
            ('propeller', 'blade'): 0.8,
            ('wheel', 'rim'): 0.7,
            ('container', 'box'): 0.6,
            ('fan', 'propeller'): 0.7,
            ('bottle', 'container'): 0.6,
            ('cup', 'container'): 0.5,
            ('sword', 'blade'): 0.5,
            ('knife', 'blade'): 0.6
        }
        
        total_score = 0
        max_scores = []
        
        for item1 in list1:
            item_scores = []
            item1_str = str(item1).lower()
            item1_words = set(item1_str.split())
            
            # Check for mechanical/automotive relevance
            mechanical_relevance = len(item1_words.intersection(mechanical_keywords)) > 0
            
            for item2 in list2:
                item2_str = str(item2).lower()
                item2_words = set(item2_str.split())
                
                # Check object type similarities for special cases
                object_pair_score = 0
                for (obj1, obj2), score in object_type_similarities.items():
                    if (obj1 in item1_str and obj2 in item2_str) or (obj2 in item1_str and obj1 in item2_str):
                        object_pair_score = score
                        break
                
                # Basic word overlap score (if words exist)
                if item1_words and item2_words:
                    overlap_score = len(item1_words.intersection(item2_words)) / len(item1_words.union(item2_words))
                else:
                    overlap_score = 0
                
                # Fuzzy match score
                fuzzy_score = fuzz.ratio(item1_str, item2_str) / 100.0
                
                # Token sort ratio for handling word order differences
                token_sort_score = fuzz.token_sort_ratio(item1_str, item2_str) / 100.0
                
                # Combined score with mechanical relevance bonus
                combined_score = max(
                    (overlap_score * 0.3 + fuzzy_score * 0.4 + token_sort_score * 0.3),
                    object_pair_score
                )
                
                if mechanical_relevance and len(item2_words.intersection(mechanical_keywords)) > 0:
                    combined_score *= 1.2  # 20% bonus for mechanical matches
                
                item_scores.append(combined_score)
            
            if item_scores:
                max_scores.append(max(item_scores))
        
        if max_scores:
            total_score = sum(max_scores) / len(max_scores)
        
        return min(total_score, 1.0)  # Cap at 1.0
    
    def calculate_complexity_with_metadata(self, code: str, metadata: Dict) -> float:
        """
        Calculate complexity score considering both code and metadata.
        Enhanced version that takes metadata into account for complexity scoring.
        """
        # Get base complexity score from code analysis
        score = self._calculate_complexity_score(code)
        
        # Metadata-based complexity factors
        try:
            # Consider declared complexity
            if metadata.get('properties_complexity') == 'intricate':
                score += 15
            elif metadata.get('properties_complexity') == 'complex':
                score += 10
            elif metadata.get('properties_complexity') == 'moderate':
                score += 5
            
            # Consider printability
            if metadata.get('properties_printability') == 'challenging':
                score += 10
            elif metadata.get('properties_printability') == 'requires_support':
                score += 5
            
            # Consider support needed
            if metadata.get('properties_support_needed') == 'extensive':
                score += 10
            elif metadata.get('properties_support_needed') == 'moderate':
                score += 5
            
        except Exception as e:
            print(f"Error calculating metadata complexity: {str(e)}")
        
        # Normalize score to 0-100 range
        normalized_score = min(max(score, 0), 100)
        return normalized_score

    def validate_metadata(self, metadata: Dict) -> bool:
        """Validate the metadata structure."""
        required_fields = [
            'object_type',
            'dimensions',
            'features',
            'materials',
            'complexity',
            'style',
            'use_case',
            'geometric_properties',
            'technical_requirements',
            'step_back_analysis'
        ]

        # Check top-level required fields
        for field in required_fields:
            if field not in metadata:
                print(f"Missing required field: {field}")
                return False

        # Check step-back analysis structure
        step_back_fields = [
            'core_principles',  # Updated from 'principles'
            'shape_components',
            'implementation_steps'
        ]

        if 'step_back_analysis' in metadata:
            for field in step_back_fields:
                if field not in metadata['step_back_analysis']:
                    print(f"Missing required step-back analysis field: {field}")
                    return False

        return True 